"""Deep-learning baselines trained from scratch (no pre-trained encoder).

    python baselines_dl.py --arch textcnn --seed 42
    python baselines_dl.py --arch bilstm
    python baselines_dl.py --arch cnn_bilstm

  textcnn      Kim-style CNN: parallel 2/3/4/5-gram convolutions, max-over-time
  bilstm       BiLSTM with max+mean pooling
  cnn_bilstm   HYBRID DL: 1-D conv feature extractor -> BiLSTM -> attention pooling

These exist to answer "does the transformer's pre-training matter?" — they see
exactly the same GoEmotions training rows as everything else and nothing more,
so any gap to RoBERTa is attributable to the encoder.
"""
from __future__ import annotations

import argparse
import copy
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from common import Timer, load_split, save_predictions, set_seed

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")  # words + punctuation/emoticon glyphs
MAX_LEN, EMB_DIM, MIN_FREQ = 40, 200, 2
PAD, UNK = 0, 1


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    counts = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, c in counts.most_common():
        if c >= MIN_FREQ:
            vocab[tok] = len(vocab)
    return vocab


def encode(texts: list[str], vocab: dict[str, int]) -> torch.Tensor:
    out = torch.zeros(len(texts), MAX_LEN, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [vocab.get(tok, UNK) for tok in tokenize(t)][:MAX_LEN]
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_labels, kernels=(2, 3, 4, 5), filters=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(EMB_DIM, filters, k, padding=k // 2) for k in kernels)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(kernels), n_labels)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)
        pooled = [F.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.out(self.drop(torch.cat(pooled, 1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_labels, hidden=192, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD)
        self.emb_drop = nn.Dropout(0.25)
        self.lstm = nn.LSTM(EMB_DIM, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 4, n_labels)

    def forward(self, ids):
        mask = (ids != PAD).unsqueeze(-1)
        h, _ = self.lstm(self.emb_drop(self.emb(ids)))
        mx = h.masked_fill(~mask, -1e4).max(1).values
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.out(self.drop(torch.cat([mx, mean], 1)))


class CNNBiLSTM(nn.Module):
    """Conv layer extracts local n-gram features; BiLSTM models their order;
    additive attention decides which positions carry the emotion."""

    def __init__(self, vocab_size, n_labels, filters=192, hidden=160, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD)
        self.emb_drop = nn.Dropout(0.25)
        self.conv = nn.Conv1d(EMB_DIM, filters, 3, padding=1)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, n_labels)

    def forward(self, ids):
        mask = ids != PAD
        x = F.relu(self.conv(self.emb_drop(self.emb(ids)).transpose(1, 2))).transpose(1, 2)
        h, _ = self.lstm(x)
        score = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4)
        ctx = (torch.softmax(score, 1).unsqueeze(-1) * h).sum(1)
        return self.out(self.drop(ctx))


ARCHS = {"textcnn": TextCNN, "bilstm": BiLSTM, "cnn_bilstm": CNNBiLSTM}
FAMILY = {"textcnn": "Deep learning", "bilstm": "Deep learning", "cnn_bilstm": "Hybrid DL"}


@torch.no_grad()
def predict(model, X, device, bs=512) -> np.ndarray:
    model.eval()
    return np.concatenate(
        [torch.sigmoid(model(X[i : i + bs].to(device))).cpu().numpy() for i in range(0, len(X), bs)]
    )


def macro_ap(Y, P) -> float:
    cols = [j for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in cols]))


def main(arch: str, seed: int, epochs: int, patience: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_x, Ytr = load_split("train")
    va_x, Yva = load_split("validation")
    te_x, _ = load_split("test")

    vocab = build_vocab(tr_x)
    Xtr, Xva, Xte = (encode(t, vocab) for t in (tr_x, va_x, te_x))
    Ytr_t = torch.from_numpy(Ytr)
    print(f"{arch}: vocab {len(vocab):,}, device {device}")

    model = ARCHS[arch](len(vocab), Ytr.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)

    best_ap, best_state, bad = -1.0, None, 0
    with Timer() as t:
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(Xtr), generator=g)
            total = 0.0
            for i in range(0, len(perm), 128):
                idx = perm[i : i + 128]
                loss = F.binary_cross_entropy_with_logits(model(Xtr[idx].to(device)), Ytr_t[idx].to(device))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            val_ap = macro_ap(Yva, predict(model, Xva, device))
            print(f"  epoch {ep:2d}  train loss {total / len(perm):.4f}  val macro PR-AUC {val_ap:.4f}")
            if val_ap > best_ap:
                best_ap, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state)

    with Timer() as t_inf:
        test_probs = predict(model, Xte, device)
    val_probs = predict(model, Xva, device)
    save_predictions(
        arch, seed, val_probs, test_probs,
        {"family": FAMILY[arch], "n_params": n_params, "train_seconds": t.seconds,
         "infer_ms_per_1k": t_inf.seconds / len(Xte) * 1e6, "val_macro_pr_auc": best_ap},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()
    main(a.arch, a.seed, a.epochs, a.patience)
