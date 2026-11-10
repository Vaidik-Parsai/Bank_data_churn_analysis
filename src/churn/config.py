"""Configuration definitions for the bank churn prediction project."""

from pathlib import Path
from typing import Final

# -----------------------------------------------------------------------------
# Base Directories
# -----------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

# -----------------------------------------------------------------------------
# Data Paths
# -----------------------------------------------------------------------------
RAW_DATA_PATH: Final[Path] = DATA_DIR / "churn_raw.csv"
CLEANED_DATA_PATH: Final[Path] = DATA_DIR / "churn_cleaned.csv"

# -----------------------------------------------------------------------------
# Model & Evaluation Paths
# -----------------------------------------------------------------------------
MODEL_OUTPUT_PATH: Final[Path] = MODELS_DIR / "churn_rf_pipeline.joblib"
METADATA_OUTPUT_PATH: Final[Path] = MODELS_DIR / "metadata.json"

# -----------------------------------------------------------------------------
# Column Schemas
# -----------------------------------------------------------------------------
TARGET_COL: Final[str] = "Exited"

# Columns to strictly drop (identifiers / high-cardinality leakage risk)
IDENTIFIER_COLS: Final[list[str]] = [
    "RowNumber",
    "CustomerId",
    "Surname",
]

# Optional EDA-engineered bucket columns to ignore during ML pipeline ingestion
EDA_BUCKET_COLS: Final[list[str]] = [
    "AgeGroup",
    "BalanceCategory",
    "TenureGroup",
]

NUMERIC_FEATURES: Final[list[str]] = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "Geography",
    "Gender",
]

FEATURE_COLUMNS: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# -----------------------------------------------------------------------------
# Model Hyperparameters & Training Defaults
# -----------------------------------------------------------------------------
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
CV_FOLDS: Final[int] = 5
SCORING_METRIC: Final[str] = "roc_auc"
DEFAULT_DECISION_THRESHOLD: Final[float] = 0.35
