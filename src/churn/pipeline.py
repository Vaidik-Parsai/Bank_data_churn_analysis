"""Scikit-learn preprocessing and model pipeline construction.

Logic extracted and formalized from exploratory notebook:
`learning_notes/01_supervised_models_and_tuning.ipynb`:
- Cell 6: Categorical mapping/encoding for 'Gender' and 'Geography'
- Cell 7: Feature isolation and identifier column exclusions
- Cell 10: StandardScaler transformation
- Cell 23 & 26: RandomForestClassifier initialization and hyperparameter configuration
"""

from typing import List, Optional

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical feature branches.

    The numeric branch imputes missing values with median and scales features
    with StandardScaler. The categorical branch imputes missing values with the
    most frequent category and encodes with OneHotEncoder (handle_unknown='ignore').

    Args:
        numeric_features: List of numeric column names. Defaults to config.NUMERIC_FEATURES.
        categorical_features: List of categorical column names. Defaults to config.CATEGORICAL_FEATURES.

    Returns:
        ColumnTransformer: Preprocessor ready to be used inside an sklearn Pipeline.
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = (
        categorical_features
        if categorical_features is not None
        else CATEGORICAL_FEATURES
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_pipeline(
    classifier: Optional[BaseEstimator] = None,
    random_state: int = RANDOM_STATE,
    n_jobs: Optional[int] = None,
) -> Pipeline:
    """Construct the end-to-end churn prediction Pipeline.

    Accepts raw customer DataFrame rows, runs preprocessing, and feeds
    features into the classifier. Defaults to RandomForestClassifier
    with class_weight='balanced' to handle class imbalance.

    Args:
        classifier: Sklearn classifier estimator. If None, instantiates a
            RandomForestClassifier(random_state=random_state, class_weight='balanced').
        random_state: Random state seed.
        n_jobs: Number of parallel jobs for estimators. Defaults to None (single thread)
            for predictable latency and compatibility.

    Returns:
        Pipeline: End-to-end Pipeline([('preprocess', ...), ('model', ...)])
    """
    preprocessor = build_preprocessor()

    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )

    return pipeline


def get_feature_names_after_preprocessing(pipeline: Pipeline) -> List[str]:
    """Retrieve human-readable feature names output by the preprocessor.

    Args:
        pipeline: Fitted scikit-learn Pipeline containing a 'preprocess' step.

    Returns:
        List[str]: Names of all transformed features (scaled numerics and OHE categories).
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    return list(feature_names)
