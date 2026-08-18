from pathlib import Path

import joblib
import os
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train import FEATURE_COLUMNS, TARGET_COLUMN


ROOT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT_DIR / ".cache"))

import matplotlib.pyplot as plt

MODELS_DIR = ROOT_DIR / "models"
DEFAULT_TEST_DATA_PATH = ROOT_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

METRIC_COLUMNS = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
CLASS_LABELS = ["Non-Spam (0)", "Spam (1)"]


@st.cache_resource
def load_models() -> dict[str, object]:
    models = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        models[model_name] = joblib.load(model_path)
    return models


@st.cache_data
def load_default_test_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_TEST_DATA_PATH)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The uploaded file could not be parsed as a valid CSV.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded CSV must be a readable text file.") from exc


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        raise ValueError("The active dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"The dataset must include the target column `{TARGET_COLUMN}`.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        missing = ", ".join(missing_features[:8])
        suffix = "..." if len(missing_features) > 8 else ""
        raise ValueError(f"Missing required feature columns: {missing}{suffix}")

    x = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    try:
        x = x.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All model feature columns must contain numeric values only.") from exc

    if x.isna().any().any():
        raise ValueError("Feature columns contain missing values. Please upload complete test data.")

    try:
        y = pd.to_numeric(y, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"The `{TARGET_COLUMN}` column must contain only 0 and 1 values.") from exc

    if y.isna().any():
        raise ValueError(f"The `{TARGET_COLUMN}` column contains missing values.")

    observed_targets = set(y.unique())
    if not observed_targets.issubset({0, 1}):
        raise ValueError(f"The `{TARGET_COLUMN}` column must contain only 0 and 1 values.")

    if observed_targets != {0, 1}:
        raise ValueError(
            "The is_spam column must contain both classes 0 and 1 to calculate all "
            "evaluation metrics, including AUC."
        )

    return x, y.astype(int)


def evaluate_model(model, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x)
    y_score = model.predict_proba(x)[:, 1]

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def build_comparison(models: dict[str, object], x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_FILES:
        metrics = evaluate_model(models[model_name], x, y)
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def render_metric_grid(metrics: dict[str, float]) -> None:
    first_row = st.columns(3)
    second_row = st.columns(3)

    for column, metric_name in zip(first_row + second_row, METRIC_COLUMNS[1:]):
        column.metric(metric_name, f"{metrics[metric_name]:.4f}")


def plot_confusion_matrix(y: pd.Series, y_pred) -> plt.Figure:
    matrix = confusion_matrix(y, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_LABELS, rotation=20, ha="right")
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_dataset_summary(x: pd.DataFrame, y: pd.Series) -> None:
    spam_count = int((y == 1).sum())
    non_spam_count = int((y == 0).sum())

    columns = st.columns(4)
    columns[0].metric("Test Rows", f"{len(y):,}")
    columns[1].metric("Input Features", f"{x.shape[1]:,}")
    columns[2].metric("Spam", f"{spam_count:,}")
    columns[3].metric("Non-Spam", f"{non_spam_count:,}")


def main() -> None:
    st.set_page_config(page_title="Spambase Classifier", layout="wide")

    st.title("Spambase Email Classification")
    st.write(
        "Evaluate trained machine learning classifiers on the UCI Spambase dataset. "
        "The target column is `is_spam`, where `1` means spam and `0` means non-spam."
    )

    st.sidebar.header("Evaluation Setup")
    selected_model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

    try:
        models = load_models()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if uploaded_file is None:
        st.info("Using bundled held-out test dataset: `test_data.csv`.")
        active_df = load_default_test_data()
        dataset_source = "Bundled test_data.csv"
    else:
        try:
            active_df = read_uploaded_csv(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        dataset_source = uploaded_file.name
        st.success(f"Using uploaded dataset: `{dataset_source}`.")

    try:
        x, y = validate_dataset(active_df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Active Dataset")
    st.caption(dataset_source)
    render_dataset_summary(x, y)

    with st.expander("Preview active data"):
        st.dataframe(active_df.head(10), width="stretch")

    selected_model = models[selected_model_name]
    selected_metrics = evaluate_model(selected_model, x, y)
    selected_predictions = selected_model.predict(x)

    st.subheader(f"{selected_model_name} Evaluation")
    render_metric_grid(selected_metrics)

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("#### Confusion Matrix")
        fig = plot_confusion_matrix(y, selected_predictions)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Classification Report")
        report = classification_report(
            y,
            selected_predictions,
            labels=[0, 1],
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, width="stretch")

    st.subheader("Model Comparison")
    comparison_df = build_comparison(models, x, y)
    st.dataframe(
        comparison_df.style.format({column: "{:.4f}" for column in METRIC_COLUMNS[1:]}),
        width="stretch",
    )

    best_row = comparison_df.sort_values("MCC", ascending=False).iloc[0]
    st.markdown(f"**Best model by MCC:** {best_row['Model']} with MCC `{best_row['MCC']:.4f}`.")
    st.write(
        "MCC is used as the primary comparison criterion because it accounts for true positives, "
        "true negatives, false positives, and false negatives."
    )


if __name__ == "__main__":
    main()
