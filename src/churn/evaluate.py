"""Model evaluation metrics, threshold tuning, and report visualization utilities."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure matplotlib uses a local writable non-interactive backend
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent.parent / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import FIGURES_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: True binary target labels (0 or 1).
        y_prob: Predicted positive class probabilities.
        threshold: Decision threshold for converting probabilities to binary predictions.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Evaluate metric trade-offs across candidate classification thresholds.

    Args:
        y_true: True binary target labels.
        y_prob: Predicted probabilities of churn.
        thresholds: List of thresholds to evaluate. Defaults to [0.20, 0.25, ..., 0.70].

    Returns:
        pd.DataFrame: Summary table comparing metrics per threshold.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.20, 0.75, 0.05)]

    records = [compute_metrics(y_true, y_prob, threshold=t) for t in thresholds]
    return pd.DataFrame(records)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    model_name: str = "Random Forest",
) -> Path:
    """Plot and save ROC Curve."""
    path = output_path or (FIGURES_DIR / "roc_curve.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}", fontsize=13, weight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    model_name: str = "Random Forest",
) -> Path:
    """Plot and save Precision-Recall Curve."""
    path = output_path or (FIGURES_DIR / "precision_recall_curve.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline_rate = float(np.mean(y_true))

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#2ca02c", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    plt.axhline(baseline_rate, color="#d62728", linestyle="--", lw=1.5, label=f"Baseline Churn Rate ({baseline_rate:.1%})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Coverage of Attritors)", fontsize=11)
    plt.ylabel("Precision (Accuracy of Churn Flags)", fontsize=11)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=13, weight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.35,
    output_path: Optional[Path] = None,
) -> Path:
    """Plot and save normalized Confusion Matrix heatmap."""
    path = output_path or (FIGURES_DIR / "confusion_matrix.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = ["Retained (0)", "Churned (1)"]
    annotations = np.array(
        [[f"{count}\n({ratio:.1%})" for count, ratio in zip(row_c, row_r)]
         for row_c, row_r in zip(cm, cm_norm)]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel(f"Predicted Class (Threshold = {threshold})", fontsize=11)
    plt.ylabel("Actual True Class", fontsize=11)
    plt.title("Confusion Matrix (Counts & Class Recall)", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: Optional[Path] = None,
    top_n: int = 15,
) -> Path:
    """Plot and save feature importances horizontal bar chart."""
    path = output_path or (FIGURES_DIR / "feature_importance.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    df_imp = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=True)
        .tail(top_n)
    )

    plt.figure(figsize=(8, 6))
    bars = plt.barh(df_imp["Feature"], df_imp["Importance"], color="#3b528b", edgecolor="black", alpha=0.85)
    plt.xlabel("MDI Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    plt.ylabel("Transformed Feature", fontsize=11)
    plt.title(f"Top {len(df_imp)} Feature Importances - Random Forest", fontsize=13, weight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path
