import os
from pathlib import Path
from urllib.request import urlretrieve

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

RAW_DATA_PATH = RAW_DIR / "spambase.data"
RAW_NAMES_PATH = RAW_DIR / "spambase.names"
METRICS_PATH = RESULTS_DIR / "model_metrics.csv"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"


FEATURE_COLUMNS = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_semicolon",
    "char_freq_left_parenthesis",
    "char_freq_left_bracket",
    "char_freq_exclamation",
    "char_freq_dollar",
    "char_freq_hash",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]
TARGET_COLUMN = "is_spam"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, output_path: Path) -> None:
    if output_path.exists():
        return

    print(f"Downloading {url} -> {output_path}")
    urlretrieve(url, output_path)


def load_spambase() -> pd.DataFrame:
    download_if_missing(DATA_URL, RAW_DATA_PATH)
    download_if_missing(NAMES_URL, RAW_NAMES_PATH)

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    return pd.read_csv(RAW_DATA_PATH, header=None, names=columns)


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "model",
                    DecisionTreeClassifier(random_state=RANDOM_STATE),
                )
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Naive Bayes": Pipeline([("model", GaussianNB())]),
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }


def model_filename(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\nDataset summary")
    print(f"Rows: {len(df)}")
    print(f"Input features: {len(FEATURE_COLUMNS)}")
    print(f"Target column: {TARGET_COLUMN}")
    print("Target meaning: 1 = spam, 0 = non-spam")
    print("Class distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index().to_string())
    print(f"Missing values: {int(df.isna().sum().sum())}")


def main() -> None:
    ensure_directories()

    df = load_spambase()
    print_dataset_summary(df)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(TEST_DATA_PATH, index=False)

    rows = []
    for model_name, model in build_models().items():
        print(f"\nTraining {model_name}...")
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)

        joblib.dump(model, MODELS_DIR / model_filename(model_name))

        rows.append({"Model": model_name, **metrics})

    metrics_df = pd.DataFrame(
        rows, columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\nModel comparison")
    print(metrics_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved metrics to {METRICS_PATH.relative_to(ROOT_DIR)}")
    print(f"Saved test data to {TEST_DATA_PATH.relative_to(ROOT_DIR)}")
    print(f"Saved fitted model artifacts to {MODELS_DIR.relative_to(ROOT_DIR)}/")


if __name__ == "__main__":
    main()
