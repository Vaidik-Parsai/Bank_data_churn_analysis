"""Data loading, feature validation, and stratified dataset splitting."""

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import (
    EDA_BUCKET_COLS,
    FEATURE_COLUMNS,
    IDENTIFIER_COLS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
)


def load_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load customer churn dataset from CSV.

    Args:
        filepath: Path to the CSV file. If None, defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")
    return pd.read_csv(path)


def prepare_features(
    df: pd.DataFrame,
    target_col: Optional[str] = TARGET_COL,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Clean and isolate feature columns from raw customer records.

    Explicitly drops identifier columns (RowNumber, CustomerId, Surname) and
    any EDA-specific bucket columns (AgeGroup, BalanceCategory, TenureGroup),
    ensuring only valid modeling features are retained.

    Args:
        df: Input DataFrame containing customer records.
        target_col: Name of target column if present in DataFrame.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: (X, y) where y is None if
        target_col is not present in df (e.g. during production inference).
    """
    data = df.copy()

    # Extract target if present
    y: Optional[pd.Series] = None
    if target_col and target_col in data.columns:
        y = data[target_col].astype(int)
        data = data.drop(columns=[target_col])

    # Explicitly drop identifier columns if present
    cols_to_drop = [c for c in IDENTIFIER_COLS + EDA_BUCKET_COLS if c in data.columns]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    # Ensure all expected feature columns are present
    missing_features = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing_features:
        raise ValueError(
            f"Input DataFrame is missing required feature columns: {missing_features}"
        )

    # Return only the specified feature columns in a stable order
    X = data[FEATURE_COLUMNS].copy()
    return X, y


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Extract features and target, and perform a stratified train/test split.

    Splitting is performed strictly before any fitting or scaling to guarantee
    zero data leakage between training and evaluation partitions.

    Args:
        df: Full raw customer DataFrame.
        test_size: Fraction of samples reserved for the test split.
        random_state: Random state seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    X, y = prepare_features(df, target_col=TARGET_COL)
    if y is None:
        raise ValueError("Target column 'Exited' is required for training/testing split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
