"""Training, tuning, baseline comparison, and artifact serialization workflow."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from churn.config import (
    CV_FOLDS,
    DEFAULT_DECISION_THRESHOLD,
    FIGURES_DIR,
    METADATA_OUTPUT_PATH,
    MODEL_OUTPUT_PATH,
    RANDOM_STATE,
    SCORING_METRIC,
)
from churn.data import get_train_test_split, load_data
from churn.evaluate import (
    analyze_thresholds,
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from churn.pipeline import (
    build_pipeline,
    get_feature_names_after_preprocessing,
)


def train_and_evaluate(
    tune_hyperparameters: bool = True,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> Dict[str, Any]:
    """Execute complete end-to-end model training, tuning, and evaluation.

    Args:
        tune_hyperparameters: If True, executes GridSearchCV across Random Forest
            hyperparameters. If False, trains default Random Forest pipeline.
        threshold: Decision threshold for positive class classification.

    Returns:
        Dict[str, Any]: Comprehensive summary of training run and evaluation metrics.
    """
    print("=" * 70)
    print("BANK CHURN MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load data and perform stratified split strictly BEFORE fitting
    df = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(
        df, random_state=RANDOM_STATE
    )
    print(f"Data Loaded: {len(df)} total records.")
    print(f"Train set: {X_train.shape} (Churn rate: {y_train.mean():.2%})")
    print(f"Test set:  {X_test.shape} (Churn rate: {y_test.mean():.2%})")

    # 2. Baseline Model: Logistic Regression with balanced class weighting
    print("\n--- [Step 1/4] Training Baseline Model (Logistic Regression) ---")
    baseline_lr = build_pipeline(
        classifier=LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baseline_cv_scores = cross_val_score(
        baseline_lr, X_train, y_train, cv=cv, scoring=SCORING_METRIC
    )
    print(f"Baseline Logistic Regression CV ROC-AUC: {baseline_cv_scores.mean():.4f} +/- {baseline_cv_scores.std():.4f}")

    baseline_lr.fit(X_train, y_train)
    y_prob_lr = baseline_lr.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_metrics(y_test.to_numpy(), y_prob_lr, threshold=0.5)
    print(f"Baseline Test ROC-AUC: {baseline_metrics['roc_auc']} | Recall: {baseline_metrics['recall']} | F1: {baseline_metrics['f1']}")

    # 3. Model Tuning: Random Forest Classifier
    print("\n--- [Step 2/4] Tuning Random Forest via Stratified GridSearchCV ---")
    base_rf_pipeline = build_pipeline(random_state=RANDOM_STATE)

    if tune_hyperparameters:
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, 12, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
        total_fits = np.prod([len(v) for v in param_grid.values()]) * CV_FOLDS
        print(f"Running GridSearchCV over {np.prod([len(v) for v in param_grid.values()])} candidates ({total_fits} total fits)...", flush=True)
        grid_search = GridSearchCV(
            estimator=base_rf_pipeline,
            param_grid=param_grid,
            scoring=SCORING_METRIC,
            cv=cv,
            n_jobs=2,
            verbose=1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = float(grid_search.best_score_)
        print(f"Optimal Random Forest CV ROC-AUC: {best_cv_score:.4f}", flush=True)
        print(f"Best Parameters: {best_params}", flush=True)
    else:
        best_pipeline = base_rf_pipeline
        best_pipeline.fit(X_train, y_train)
        best_params = {"default": True}
        best_cv_score = float(cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring=SCORING_METRIC).mean())

    # 4. Evaluation on Held-Out Test Set
    print("\n--- [Step 3/4] Evaluating Best Pipeline on Test Set ---")
    y_test_np = y_test.to_numpy()
    y_prob_rf = best_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate at default threshold (0.50) and calibrated threshold
    rf_metrics_default = compute_metrics(y_test_np, y_prob_rf, threshold=0.5)
    rf_metrics_calibrated = compute_metrics(y_test_np, y_prob_rf, threshold=threshold)

    print(f"\nRandom Forest Performance at default threshold (0.50):")
    print(f"  ROC-AUC:   {rf_metrics_default['roc_auc']}")
    print(f"  PR-AUC:    {rf_metrics_default['pr_auc']}")
    print(f"  Recall:    {rf_metrics_default['recall']}")
    print(f"  Precision: {rf_metrics_default['precision']}")
    print(f"  F1 Score:  {rf_metrics_default['f1']}")

    print(f"\nRandom Forest Performance at tuned threshold ({threshold}):")
    print(f"  Recall:    {rf_metrics_calibrated['recall']} (caught {rf_metrics_calibrated['true_positives']} of {y_test_np.sum()} churners)")
    print(f"  Precision: {rf_metrics_calibrated['precision']}")
    print(f"  F1 Score:  {rf_metrics_calibrated['f1']}")

    # Threshold trade-off analysis table
    threshold_df = analyze_thresholds(y_test_np, y_prob_rf)
    print("\nThreshold Selection Grid:")
    print(threshold_df[["threshold", "precision", "recall", "f1", "specificity"]].to_string(index=False))

    # Feature Importance Extraction
    transformed_features = get_feature_names_after_preprocessing(best_pipeline)
    rf_classifier = best_pipeline.named_steps["model"]
    importances = rf_classifier.feature_importances_
    feat_importance_dict = {
        name: round(float(imp), 4)
        for name, imp in zip(transformed_features, importances)
    }
    sorted_feat_importance = dict(sorted(feat_importance_dict.items(), key=lambda x: x[1], reverse=True))

    # 5. Export Visualizations to reports/figures/
    print("\n--- [Step 4/4] Exporting Evaluation Plots & Metadata ---")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(y_test_np, y_prob_rf, model_name="Random Forest Pipeline")
    plot_precision_recall_curve(y_test_np, y_prob_rf, model_name="Random Forest Pipeline")
    plot_confusion_matrix(y_test_np, y_prob_rf, threshold=threshold)
    plot_feature_importance(transformed_features, importances)
    print(f"Saved evaluation plots to: {FIGURES_DIR.resolve()}")

    # 6. Save Fitted Pipeline and Metadata
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    print(f"Saved fitted pipeline to: {MODEL_OUTPUT_PATH.resolve()}")

    metadata: Dict[str, Any] = {
        "model_name": "Bank Customer Churn Predictor (Random Forest Pipeline)",
        "model_version": "1.0.0",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "python_version": "3.14",
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "primary_metric": SCORING_METRIC,
        "best_hyperparameters": best_params,
        "cross_validation_roc_auc_mean": round(best_cv_score, 4),
        "decision_threshold": threshold,
        "decision_threshold_rationale": (
            f"In customer retention, false negatives (missed attritors) have a far greater "
            f"financial impact than false positives (retention intervention costs). Lowering "
            f"the decision threshold from 0.50 to {threshold} boosts recall from "
            f"{rf_metrics_default['recall']:.1%} to {rf_metrics_calibrated['recall']:.1%} "
            f"while maintaining defensible precision ({rf_metrics_calibrated['precision']:.1%})."
        ),
        "baseline_comparison": {
            "model": "LogisticRegression(class_weight='balanced')",
            "test_metrics": baseline_metrics,
        },
        "test_metrics_default_threshold": rf_metrics_default,
        "test_metrics_calibrated_threshold": rf_metrics_calibrated,
        "features": {
            "preprocessed_features": transformed_features,
            "feature_importances": sorted_feat_importance,
        },
    }

    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {METADATA_OUTPUT_PATH.resolve()}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate(tune_hyperparameters=True)
