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

| Dataset | Used | Role | Link |
|---------|------|------|------|
| GoEmotions | Yes (main) | Train/evaluate the RoBERTa emotion classifier (27 emotions + neutral) | https://huggingface.co/datasets/google-research-datasets/go_emotions |
| dair-ai/emotion (CARER) | Yes (main) | Auxiliary corpus for multi-corpus training (masked multi-label loss) | https://huggingface.co/datasets/dair-ai/emotion |
| DailyDialog | Yes (main) | Auxiliary corpus for multi-corpus training (masked multi-label loss) | https://huggingface.co/datasets/li2017dailydialog/daily_dialog |
| Amazon Polarity | Yes (main) | Inference/demo only on real product reviews, never used for fine-tuning | https://huggingface.co/datasets/fancyzhx/amazon_polarity |
| ISEAR | No — related work | Emotion antecedents survey dataset (7 classes), referenced in literature survey | https://huggingface.co/datasets/TahaRasouli/ISEAR |
| SemEval-2018 Task 1 (Affect in Tweets) | No — related work | Multi-label tweet emotion intensity/classification, considered alternative corpus | https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2018_task_1 |
| SemEval-2019 Task 3 (EmoContext) | No — related work | Contextual emotion detection in text, referenced in literature survey | https://aclanthology.org/S19-2005/ |
| SemEval-2014 Task 4 (ABSA) | No — related work | Aspect-based sentiment analysis (restaurant/laptop reviews), motivated segment-level design | https://alt.qcri.org/semeval2014/task4/ |
| Yelp Polarity | No — related work | Large-scale binary review sentiment, considered alternative demo corpus | https://huggingface.co/datasets/fancyzhx/yelp_polarity |
| IMDB Large Movie Review Dataset | No — related work | Binary movie review sentiment, considered alternative demo corpus | https://huggingface.co/datasets/stanfordnlp/imdb |

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
