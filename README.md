# BITS Pilani Machine Learning Assignment 2

## Problem Statement

This project builds a binary classification baseline for detecting spam emails using the UCI Spambase dataset.

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
