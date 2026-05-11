"""
Evaluation utilities for AccidentXAI.

This module handles:
- classification metrics
- confusion matrix
- ROC-AUC
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt


def evaluate_model(
    y_true,
    y_pred,
    y_prob,
    model_name="Model"
):
    """
    Print evaluation metrics.
    """

    print("=" * 60)

    print(f"{model_name} Evaluation")

    print("=" * 60)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    print(f"\nROC-AUC Score: {roc_auc:.4f}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name="Model"
):
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.colorbar()

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.show()