# Response to reviewer comment

> **Comment.** The single-model baseline is not a suitable ML baseline. Explore
> hybrid architectures (TF-IDF + ML with BERT/LSTM/CNN, hybrid ML ensembles,
> CNN+LSTM, BERT+LSTM, Transformer+ML) and include more evaluation metrics
> (AUC-ROC, specificity, sensitivity, MCC, Cohen's kappa, PR-AUC, log loss, ...),
> with a comparative analysis showing whether the proposed hybrid gives a
> meaningful improvement.

We agree with the premise. The earlier evaluation compared one RoBERTa model
against itself with and without auxiliary data, which says little about whether
a transformer is needed at all. We have replaced it with a 13-model study, an
extended metric suite and significance testing. The result is more nuanced than
"the hybrid wins", and we report it that way.

## Point-by-point

| # | Reviewer request | What we did | Where |
|---|---|---|---|
| 1 | A suitable ML baseline | Added four TF-IDF classical models (Logistic Regression, Linear SVM, Naive Bayes, Random Forest; word 1-2gram + char 2-5gram features, one-vs-rest) | `src/ml/baselines_ml.py` |
| 2 | Hybrid ML ensembles | TF-IDF ML ensemble: soft vote over LR + SVM + RF (NB dropped by a validation PR-AUC quality floor) | `baselines_ml.py` |
| 3 | CNN / LSTM / CNN+LSTM | TextCNN, BiLSTM, CNN-BiLSTM-Attention trained from scratch, 3 seeds each | `src/ml/baselines_dl.py` |
| 4 | BERT / BERT+LSTM | Fine-tuned BERT-base (1 seed), RoBERTa-base (3 seeds), RoBERTa-BiLSTM-Attention (3 seeds; token states -> BiLSTM -> attention + max pool + `<s>`) | `src/ml/train_transformer.py` |
| 5 | TF-IDF/Transformer + ML | RoBERTa + TF-IDF ML fusion, and the proposed RoBERTa-BiLSTM-Attention + TF-IDF ML: late fusion `p = (1-w)·p_neural + w·p_ml`, one global `w` per seed picked on validation | `src/ml/fusion.py` |
| 6 | More metrics | AUC-ROC (macro/micro), PR-AUC (macro/micro), sensitivity, specificity, balanced accuracy, MCC (macro/micro), Cohen's kappa, F1 (macro/micro/weighted/samples), precision, Jaccard, exact-match, top-1, Hamming loss, log loss, Brier, ECE, coverage error, ranking loss, label-ranking AP, bucket-level F1/AUC | `src/ml/metrics.py` |
| 7 | Fair comparison | Every model emits probabilities only; one scorer; per-label thresholds tuned on validation for all models; same split | `src/ml/compare.py` |
| 8 | Is the improvement meaningful? | Bootstrap 95% CIs and paired-bootstrap tests of the proposed model against every other model; cost table (parameters, train/inference time) | `results/comparison/significance.csv`, `confidence_intervals.csv`, `cost.csv` |
| 9 | Reproducibility | One resumable script reproduces everything (~2.5 h on an RTX 3060 6GB) | `src/ml/run_experiments.sh` |

## Is the proposed hybrid a meaningful improvement?

Test set: GoEmotions, 5,427 comments, 28 labels. Means over 3 seeds where available.

| Model | Macro F1 | Macro MCC | Macro AUC-ROC | Macro PR-AUC | Log loss |
|---|---|---|---|---|---|
| Best classical (TF-IDF SVM) | 0.4883 | 0.4739 | 0.9040 | 0.4610 | 0.0938 |
| TF-IDF ML ensemble | 0.4849 | 0.4704 | 0.9082 | 0.4616 | 0.0937 |
| Best from-scratch DL (BiLSTM) | 0.4810 | 0.4691 | 0.8984 | 0.4485 | 0.0996 |
| RoBERTa-base | 0.5389 | 0.5276 | 0.9406 | 0.5432 | 0.0802 |
| RoBERTa + TF-IDF fusion | 0.5519 | 0.5400 | 0.9452 | 0.5485 | 0.0800 |
| **Proposed hybrid** | 0.5460 | 0.5350 | 0.9451 | 0.5526 | 0.0797 |

1. **Versus classical ML, hybrid ML and from-scratch deep models: a clear,
   significant improvement** (+0.06 to +0.25 macro F1, +0.04 to +0.14 macro
   AUC-ROC; all paired-bootstrap CIs exclude zero). This answers the reviewer's
   concern: a single classical model is indeed weak, and the gap to the
   proposed approach is large.
2. **That gain comes mostly from the pretrained transformer.** Plain fine-tuned
   BERT-base (0.5448) and RoBERTa-base (0.5389) already capture it. From-scratch
   CNN/LSTM hybrids do not beat the TF-IDF models, and the TF-IDF ML ensemble
   does not beat the SVM on macro F1.
3. **Versus plain BERT/RoBERTa the proposed hybrid is not significantly better**
   on macro F1, micro F1, MCC or PR-AUC (CIs include zero). It has a small but
   significant AUC-ROC edge (+0.007 vs. BERT-base, +0.005 vs. RoBERTa-base).
4. **Versus RoBERTa + TF-IDF fusion it is not better.** No significant difference
   on any tested metric, and the fusion model has the higher mean macro F1,
   MCC and kappa. The simpler fusion is therefore the more defensible choice on
   parsimony; the BiLSTM-attention head shows no measurable added value once the
   TF-IDF branch is present. Versus RoBERTa-BiLSTM-Attention alone, only
   AUC-ROC differs (+0.008).
5. Where the proposed model is numerically best (micro AUC-ROC, macro PR-AUC,
   coverage error, ranking loss, log loss) the margins are in the third or fourth
   decimal place and are not individually tested for significance.
6. The TF-IDF branch helps most on rare labels (e.g. relief 0.341 -> 0.515,
   grief 0.458 -> 0.548 for RoBERTa -> RoBERTa + fusion), but those labels have
   6-11 test examples, so we treat this as suggestive.

**Our conclusion:** the hybrid-versus-single-baseline concern is resolved in
favour of transformer-based models, but we do not claim that the proposed
BiLSTM + TF-IDF hybrid beats a fine-tuned RoBERTa (or RoBERTa + TF-IDF fusion)
in accuracy. The project's contribution stands on its end-to-end capabilities
(token-level SHAP explanations, masked multi-corpus training, segment-level
scoring, business buckets, deployed API/dashboard) with the comparison showing
transformers are warranted and hybridisation is roughly accuracy-neutral at a
~2% parameter and ~20% training-time cost.

## Limitations of this evidence

- Significance tests use one seed's predictions and resample test rows only (300
  resamples); they ignore training-seed variance (macro-F1 std 0.002-0.008) and
  apply no multiple-comparison correction.
- BERT-base and the classical models are single runs.
- The fusion weight varied across seeds (proposed model: 0.05, 0.50, 0.35).
- All numbers are on GoEmotions (Reddit comments); no labelled product-review set
  exists, so deployment-domain accuracy is unmeasured.
- The proposed hybrid is evaluated from saved probabilities and is not a deployed
  checkpoint; the API serves `model-v1` (plain RoBERTa, older training pipeline,
  macro F1 0.475, not comparable to the 0.539 of the comparison pipeline's
  RoBERTa-base).

## Pointers

- Full tables: [`results/comparison/RESULTS.md`](../results/comparison/RESULTS.md)
- Analysis and threats to validity: [`COMPARISON.md`](COMPARISON.md)
- Model specification: [`MODEL_CARD.md`](MODEL_CARD.md)
- Reproduce: `cd src/ml && bash run_experiments.sh`
