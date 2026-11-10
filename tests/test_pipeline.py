"""Tests for data preparation, feature extraction, and pipeline transformations."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    IDENTIFIER_COLS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
)
from churn.data import get_train_test_split, prepare_features
from churn.pipeline import (
    build_pipeline,
    build_preprocessor,
    get_feature_names_after_preprocessing,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture providing a synthetic raw DataFrame resembling churn_raw.csv."""
    n_samples = 100
    np.random.seed(42)
    return pd.DataFrame(
        {
            "RowNumber": range(1, n_samples + 1),
            "CustomerId": [15600000 + i for i in range(n_samples)],
            "Surname": [f"Surname_{i}" for i in range(n_samples)],
            "CreditScore": np.random.randint(400, 850, size=n_samples),
            "Geography": np.random.choice(["France", "Germany", "Spain"], size=n_samples),
            "Gender": np.random.choice(["Male", "Female"], size=n_samples),
            "Age": np.random.randint(18, 75, size=n_samples),
            "Tenure": np.random.randint(0, 10, size=n_samples),
            "Balance": np.random.uniform(0, 200000, size=n_samples),
            "NumOfProducts": np.random.randint(1, 4, size=n_samples),
            "HasCrCard": np.random.choice([0, 1], size=n_samples),
            "IsActiveMember": np.random.choice([0, 1], size=n_samples),
            "EstimatedSalary": np.random.uniform(20000, 150000, size=n_samples),
            "Exited": np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
            # EDA bucket columns that might appear in cleaned datasets:
            "AgeGroup": ["31-40"] * n_samples,
            "BalanceCategory": ["High"] * n_samples,
            "TenureGroup": ["3-5"] * n_samples,
        }
    )


def test_prepare_features_drops_identifiers_and_eda_columns(sample_raw_dataframe):
    """Ensure identifier and EDA bucket columns are strictly dropped."""
    X, y = prepare_features(sample_raw_dataframe, target_col=TARGET_COL)

    assert y is not None
    assert len(y) == len(sample_raw_dataframe)
    for id_col in IDENTIFIER_COLS:
        assert id_col not in X.columns
    for eda_col in ["AgeGroup", "BalanceCategory", "TenureGroup"]:
        assert eda_col not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS


def test_get_train_test_split_stratification(sample_raw_dataframe):
    """Ensure train/test split maintains target class distribution."""
    X_train, X_test, y_train, y_test = get_train_test_split(
        sample_raw_dataframe, test_size=0.2, random_state=RANDOM_STATE
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    # Stratification check: proportion of churn should be closely matched
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.10


def test_pipeline_fit_predict_raw_dataframe(sample_raw_dataframe):
    """Verify end-to-end pipeline fits and predicts on raw DataFrame rows."""
    X_train, X_test, y_train, y_test = get_train_test_split(
        sample_raw_dataframe, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)

    # Test single-row inference (as in a REST API request)
    single_row = X_test.iloc[0:1]
    prob = pipeline.predict_proba(single_row)
    pred = pipeline.predict(single_row)

    assert prob.shape == (1, 2)
    assert np.isclose(prob.sum(), 1.0)
    assert pred[0] in [0, 1]


def test_pipeline_handles_unseen_categories(sample_raw_dataframe):
    """Ensure OneHotEncoder handle_unknown='ignore' works for unseen categorical values."""
    X_train, X_test, y_train, _ = get_train_test_split(
        sample_raw_dataframe, test_size=0.2, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline(random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)

    unseen_row = X_test.iloc[0:1].copy()
    unseen_row["Geography"] = "NonExistentCountry"
    unseen_row["Gender"] = "Other"

    # Should not raise an error; unobserved categories become all-zero one-hot vectors
    probs = pipeline.predict_proba(unseen_row)
    assert probs.shape == (1, 2)


def test_cross_validation_wraps_pipeline(sample_raw_dataframe):
    """Verify CV wraps the entire pipeline (imputation + scaling + model)."""
    X, y = prepare_features(sample_raw_dataframe)
    pipeline = build_pipeline(random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
