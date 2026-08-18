# BITS Pilani Machine Learning Assignment 2

## Problem Statement

This project builds a binary classification baseline for detecting spam emails using the UCI Spambase dataset.

## GitHub Repository

Repository URL: https://github.com/IshanChow/mfmsl_asnt2

## Live Streamlit Application

Live URL: https://mfmslasnt2-gbhgdxbckplxgmz3rk6tnr.streamlit.app/

## Dataset

- Name: UCI Spambase
- Source: https://archive.ics.uci.edu/ml/datasets/spambase
- Raw data URL: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
- Dimensions: 4,601 rows and 57 input features
- Target column: `is_spam`
- Target meaning: `1` = spam, `0` = non-spam

## Models Implemented

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Gaussian Naive Bayes Classifier
- Random Forest Classifier

## Setup and Training

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```

The training script downloads the raw Spambase data into `data/raw/` if it is not already present, creates a stratified 80/20 train/test split, saves fitted model artifacts under `models/`, writes held-out test data to `test_data.csv`, and saves metrics to `results/model_metrics.csv`.

## Streamlit Application

Launch the app:

```bash
streamlit run app.py
```

The app loads the fitted model artifacts from `models/` and evaluates them without retraining. It supports:

- CSV upload using the same schema as `test_data.csv`
- bundled `test_data.csv` evaluation when no file is uploaded
- model selection across all five classifiers
- Accuracy, AUC, Precision, Recall, F1, and MCC metrics
- confusion matrix
- classification report
- all-model comparison table for the active dataset

The uploaded or bundled CSV must include all 57 Spambase input feature columns and the true target column `is_spam`.

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.929425 | 0.970196 | 0.920904 | 0.898072 | 0.909344 | 0.851761 |
| Decision Tree | 0.910966 | 0.907751 | 0.882834 | 0.892562 | 0.887671 | 0.813967 |
| KNN | 0.907709 | 0.950564 | 0.886111 | 0.878788 | 0.882434 | 0.806495 |
| Naive Bayes | 0.833876 | 0.944926 | 0.717842 | 0.953168 | 0.818935 | 0.694114 |
| Random Forest | 0.945711 | 0.983385 | 0.948424 | 0.911846 | 0.929775 | 0.886010 |

## Model Observations

- Logistic Regression: Strong overall performance with high AUC and balanced precision/recall. It performs surprisingly close to the best ensemble model despite being relatively simple.
- Decision Tree: Good classification performance but lower AUC and MCC than Logistic Regression and Random Forest, indicating weaker generalization on the held-out data.
- KNN: Competitive performance with good AUC, but slightly lower Accuracy, F1 and MCC than Logistic Regression and Random Forest.
- Naive Bayes: Achieves the highest recall, meaning it identifies most spam messages, but substantially lower precision indicates more false-positive spam classifications.
- Random Forest: Best overall model, with the highest Accuracy, AUC, Precision, F1 and MCC, showing the strongest overall discrimination and balanced classification performance.

## Overall Winner

Random Forest is the overall winning model based primarily on MCC. It also leads most other metrics, including Accuracy, AUC, Precision, and F1.
