"""Transformer baselines and the proposed RoBERTa-BiLSTM hybrid, one code path.

    python train_transformer.py --arch plain  --encoder roberta-base --seed 42   # transformer baseline
    python train_transformer.py --arch plain  --encoder bert-base-uncased        # second transformer baseline
    python train_transformer.py --arch hybrid --encoder roberta-base --seed 42   # proposed neural core

  plain    encoder -> <s>/[CLS] state -> dense+tanh -> linear      (standard fine-tuning head)
  hybrid   encoder -> token states -> BiLSTM -> [attention pool ; max pool ; <s>] -> linear

Why the hybrid head: the [CLS] vector must compress a whole comment into one
position. GoEmotions comments frequently carry two emotions at different spans
("thanks, but this is awful"), and a recurrent layer over the token states with
attention pooling lets each label attend to the span that evidences it, while
the max-pool and <s> paths keep the strong global signal the plain head already
has. It adds ~2.4M parameters (~2%) to the encoder.

Both arches use the same loss, optimiser, schedule, early-stopping metric and
data as every other model in the comparison. Only the probabilities are saved;
compare.py does all scoring.
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import Timer, load_split, save_predictions, set_seed

MAX_LEN = 64  # 99th percentile is 27 words; 64 subword tokens truncates <0.1% of rows


class EmotionModel(nn.Module):
    def __init__(self, encoder_name: str, n_labels: int, arch: str, lstm_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.arch = arch
        self.encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False)
        d = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        if arch == "plain":
            self.dense = nn.Linear(d, d)
            self.out = nn.Linear(d, n_labels)
        else:
            self.lstm = nn.LSTM(d, lstm_hidden, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(2 * lstm_hidden, 1)
            self.out = nn.Linear(2 * (2 * lstm_hidden) + d, n_labels)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = h[:, 0]
        if self.arch == "plain":
            return self.out(self.drop(torch.tanh(self.dense(self.drop(cls)))))
        mask = attention_mask.bool()
        # Length-aware packing keeps the backward LSTM from reading pad tokens.
        lengths = attention_mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(h).float(), lengths, batch_first=True, enforce_sorted=False)
        seq, _ = self.lstm(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True, total_length=h.size(1))
        score = self.attn(seq).squeeze(-1).masked_fill(~mask, -1e4)
        attn_pool = (torch.softmax(score, 1).unsqueeze(-1) * seq).sum(1)
        max_pool = seq.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        feats = torch.cat([attn_pool, max_pool, cls.float()], dim=1)
        return self.out(self.drop(feats))


def batches(enc, Y, idx, bs):
    for i in range(0, len(idx), bs):
        sl = idx[i : i + bs]
        ids = pad_sequence([torch.tensor(enc[j]) for j in sl], batch_first=True, padding_value=enc.pad_id)
        yield ids, (ids != enc.pad_id).long(), (None if Y is None else torch.from_numpy(Y[sl]))


class Encoded:
    def __init__(self, tokenizer, texts):
        self.ids = tokenizer(texts, truncation=True, max_length=MAX_LEN)["input_ids"]
        self.pad_id = tokenizer.pad_token_id

    def __getitem__(self, i):
        return self.ids[i]

    def __len__(self):
        return len(self.ids)


@torch.no_grad()
def predict(model, enc, device, bs=128) -> np.ndarray:
    model.eval()
    order = np.argsort([len(x) for x in enc.ids])  # length-sorted -> minimal padding
    out = np.zeros((len(enc), model.out.out_features), dtype=np.float32)
    for pos, (ids, mask, _) in zip(range(0, len(order), bs), batches(enc, None, order, bs)):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(ids.to(device), mask.to(device))
        out[order[pos : pos + bs]] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def macro_ap(Y, P) -> float:
    cols = [j for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in cols]))


def main(a) -> None:
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.encoder)
    tr_x, Ytr = load_split("train")
    va_x, Yva = load_split("validation")
    te_x, _ = load_split("test")
    Etr, Eva, Ete = Encoded(tok, tr_x), Encoded(tok, va_x), Encoded(tok, te_x)

    model = EmotionModel(a.encoder, Ytr.shape[1], a.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.head_parameters())
    print(f"{a.arch}/{a.encoder}: {n_params/1e6:.1f}M params ({head_params/1e6:.2f}M in head), device {device}")

    # Encoder gets the usual small fine-tuning LR; freshly initialised head layers train faster.
    opt = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": a.lr},
         {"params": model.head_parameters(), "lr": a.head_lr}],
        weight_decay=0.01,
    )
    steps = a.epochs * int(np.ceil(len(Etr) / a.batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    rng = np.random.default_rng(a.seed)

    best_ap, best_state, bad = -1.0, None, 0
    with Timer() as t:
        for ep in range(1, a.epochs + 1):
            model.train()
            perm = rng.permutation(len(Etr))
            total = 0.0
            for ids, mask, y in batches(Etr, Ytr, perm, a.batch_size):
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(ids.to(device), mask.to(device))
                loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * len(y)
            val_ap = macro_ap(Yva, predict(model, Eva, device))
            print(f"  epoch {ep}  train loss {total/len(Etr):.4f}  val macro PR-AUC {val_ap:.4f}  ({t_elapsed(t)})", flush=True)
            if val_ap > best_ap:
                best_ap, bad = val_ap, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
    model.load_state_dict(best_state)

    val_probs = predict(model, Eva, device)
    with Timer() as t_inf:
        test_probs = predict(model, Ete, device)
    name = a.name or (f"{a.encoder.split('/')[-1].replace('-base', '').replace('-uncased', '')}"
                      f"_{'bilstm_attn' if a.arch == 'hybrid' else 'plain'}")
    save_predictions(
        name, a.seed, val_probs, test_probs,
        {"family": "Hybrid DL (Transformer+BiLSTM)" if a.arch == "hybrid" else "Transformer",
         "encoder": a.encoder, "arch": a.arch, "n_params": n_params, "head_params": head_params,
         "train_seconds": t.seconds, "infer_ms_per_1k": t_inf.seconds / len(Ete) * 1e6,
         "val_macro_pr_auc": best_ap, "epochs_run": ep},
    )


def t_elapsed(t: Timer) -> str:
    import time
    return f"{time.perf_counter() - t.t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["plain", "hybrid"], required=True)
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--name", default=None)
    main(ap.parse_args())
