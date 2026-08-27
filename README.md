# Emotion-Aware Feedback Analysis System

Explainable Transformer-Based Emotion-Aware Customer Feedback Analysis
*A Multi-Label Fine-Grained Emotion Classification System with SHAP-Based Token Attribution for Business-Actionable Review Insights*

Course: Applied Machine Learning for Text Analysis (24ALT3101)
KLH University, Hyderabad

## Team Members

| Name | ID Number |
|------|-----------|
| Pujith Krishna Soma | 2420090069 |
| Yaswanth Chowdary | 2420090106 |
| V Sriram Subramanyam | 2420090126 |
| Ch G Susheel | 2420090124 |

## Supervisor

Ms Katanguri Swanthana

## Abstract

Conventional sentiment analysis tools classify customer feedback into coarse buckets — positive, negative, or neutral — which gives businesses no actionable signal on what kind of negative emotion is present or which phrases drove the model's decision. This project builds a transformer-based multi-label emotion classification framework with an integrated explainability layer to close three gaps in existing tools: coarseness, opacity, and mixed-sentiment blindness. A pretrained RoBERTa-base model is fine-tuned on the GoEmotions taxonomy (27 emotions + neutral, multi-label sigmoid classification) so a single review can carry more than one emotion simultaneously. Fine-grained emotions are rolled up into four business-actionable buckets (satisfaction, frustration, confusion, concern), and SHAP (SHapley Additive exPlanations) is applied at the token level so stakeholders can see exactly which words drove each prediction. Reviews are additionally split into segments and scored independently, so a review that shifts from praise to complaint is reported as two distinct signals instead of one averaged score. The result is a fine-grained, explainable emotion classifier that non-technical business stakeholders can trust and act on.

## Repository Structure

```
/src        Source code (model training, inference, SHAP explainability pipeline)
/docs       Project documentation, including the full capstone report and project abstract
/data       Datasets or documented data source references (e.g. GoEmotions)
/results    Model outputs, evaluation metrics, prediction artifacts
/reports    Generated reports, writeups, presentation material
README.md   This file
```

## Datasets

10 emotion/sentiment datasets surveyed; **4 main datasets used** in this project (marked below), rest referenced as related work / considered alternatives.

| Dataset | Used | Role | Source Paper | Link |
|---------|------|------|--------------|------|
| GoEmotions | Yes (main) | Train/evaluate the RoBERTa emotion classifier (27 emotions + neutral) | Demszky et al., 2020 — [GoEmotions: A Dataset of Fine-Grained Emotions](https://aclanthology.org/2020.acl-main.372/) (ACL 2020) | https://huggingface.co/datasets/google-research-datasets/go_emotions |
| dair-ai/emotion (CARER) | Yes (main) | Auxiliary corpus for multi-corpus training (masked multi-label loss) | Saravia et al., 2018 — [CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404/) (EMNLP 2018) | https://huggingface.co/datasets/dair-ai/emotion |
| DailyDialog | Yes (main) | Auxiliary corpus for multi-corpus training (masked multi-label loss) | Li et al., 2017 — [DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://aclanthology.org/I17-1099/) (IJCNLP 2017) | https://huggingface.co/datasets/li2017dailydialog/daily_dialog |
| Amazon Polarity | Yes (main) | Inference/demo only on real product reviews, never used for fine-tuning | Zhang, Zhao & LeCun, 2015 — [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626) (NeurIPS 2015) | https://huggingface.co/datasets/fancyzhx/amazon_polarity |
| ISEAR | No — related work | Emotion antecedents survey dataset (7 classes), referenced in literature survey | Scherer & Wallbott, 1994 — [Evidence for Universality and Cultural Variation of Differential Emotion Response Patterning](https://doi.org/10.1037/0022-3514.66.2.310) (J. Personality and Social Psychology) | https://huggingface.co/datasets/TahaRasouli/ISEAR |
| SemEval-2018 Task 1 (Affect in Tweets) | No — related work | Multi-label tweet emotion intensity/classification, considered alternative corpus | Mohammad et al., 2018 — [SemEval-2018 Task 1: Affect in Tweets](https://aclanthology.org/S18-1001/) (SemEval 2018) | https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2018_task_1 |
| SemEval-2019 Task 3 (EmoContext) | No — related work | Contextual emotion detection in text, referenced in literature survey | Chatterjee et al., 2019 — [SemEval-2019 Task 3: EmoContext Contextual Emotion Detection in Text](https://aclanthology.org/S19-2005/) (SemEval 2019) | https://aclanthology.org/S19-2005/ |
| SemEval-2014 Task 4 (ABSA) | No — related work | Aspect-based sentiment analysis (restaurant/laptop reviews), motivated segment-level design | Pontiki et al., 2014 — [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://aclanthology.org/S14-2004/) (SemEval 2014) | https://alt.qcri.org/semeval2014/task4/ |
| Yelp Polarity | No — related work | Large-scale binary review sentiment, considered alternative demo corpus | Zhang, Zhao & LeCun, 2015 — [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626) (NeurIPS 2015) | https://huggingface.co/datasets/fancyzhx/yelp_polarity |
| IMDB Large Movie Review Dataset | No — related work | Binary movie review sentiment, considered alternative demo corpus | Maas et al., 2011 — [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/) (ACL 2011) | https://huggingface.co/datasets/stanfordnlp/imdb |

### Gaps in Prior Work vs. What This Project Fulfills

| Dataset / Paper | Gap in Original Work | Fulfilled By This Project |
|---|---|---|
| GoEmotions (Demszky et al., 2020) | Reddit-domain text doesn't transfer cleanly to product-review domain; rare emotions (grief, pride, relief) have near-zero F1 due to low support | Domain shift is stated explicitly rather than hidden (Amazon reviews used only for inference, never fine-tuning); business-bucket rollup absorbs low-support labels into 4 actionable categories instead of relying on unreliable rare-label predictions |
| CARER / dair-ai emotion (Saravia et al., 2018) | Single-label only (one emotion per text); hashtag-based weak supervision, not human-annotated | Folded in only as an auxiliary corpus under masked multi-label loss — contributes signal without forcing a single-label assumption onto the multi-label model |
| DailyDialog (Li et al., 2017) | Small label set (7 emotions), heavily skewed toward "no emotion," dialogue domain differs from review/feedback text | Used only as auxiliary training signal via masked loss (only annotated label positions penalized), not as primary domain data |
| Amazon Polarity (Zhang, Zhao & LeCun, 2015) | Coarse binary polarity only — no fine-grained emotion, no explainability | Same reviews scored with 27-emotion multi-label output + SHAP token attribution + segment-level mixed-sentiment detection, instead of a single positive/negative label |
| ISEAR (Scherer & Wallbott, 1994) | Small, dated (1994), single-label, self-report survey bias | Not used for training; its coarse single-label paradigm is exactly what GoEmotions' multi-label taxonomy and this project's sigmoid classification head supersede |
| SemEval-2018 Task 1 (Mohammad et al., 2018) | Tweet-domain noise; only 11 emotion categories; no explainability layer | Broader 28-label taxonomy (27 + neutral) plus SHAP token attribution not present in the original task |
| SemEval-2019 Task 3 / EmoContext (Chatterjee et al., 2019) | Only 4 output classes (happy/sad/angry/others); context limited to fixed 3-turn conversations | Fine-grained 27-emotion output, unrestricted review length (up to 128 tokens), and explainability, instead of a fixed-turn 4-class setup |
| SemEval-2014 Task 4 / ABSA (Pontiki et al., 2014) | Requires a fixed, manually predefined aspect taxonomy per domain (restaurant/laptop); doesn't generalize to open-domain emotion labels | Segment-level scoring splits on sentence boundaries and contrastive connectives dynamically — captures intra-review sentiment shifts without any predefined aspect category |
| Yelp Polarity (Zhang, Zhao & LeCun, 2015) | Binary sentiment only; no emotion granularity; whole-document scoring hides mixed sentiment | Multi-label fine-grained emotion classification + segment-level scoring + SHAP explanation address all three gaps simultaneously |
| IMDB (Maas et al., 2011) | Single global polarity label per document; no explainability; no mixed-sentiment handling | Segment-level scoring surfaces sentiment shifts within a document instead of one averaged label; SHAP makes every prediction auditable |

## Setup and Execution Instructions

```bash
# Clone the repository
git clone https://github.com/psoma10/ALT.git
cd ALT-Project

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training / inference (scripts to be added under /src)
python src/train.py
python src/predict.py
```

*Setup steps will be updated as source code, dependencies, and data pipeline are added under `/src` and `/data`.*

## Current Phase Status

Project setup phase — repository structure and capstone project report are in place. Model implementation under `/src` is in progress.
