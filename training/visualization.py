"""
Machine Learning visualization module.

Responsibilities
----------------
- Generate ML evaluation plots
- Save visualization artifacts


This module does NOT:
- Load datasets
- Train models
- Calculate metrics
- Handle MLflow
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# ============================================================
# Output directories
# ============================================================


BASE_DIR = Path(".")


PLOT_DIR = BASE_DIR / "reports" / "evaluation" / "plots"


PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Confusion matrix
# ============================================================


def plot_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Generate confusion matrix plot.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix)

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    labels = ["Benign", "Malignant"]

    plt.xticks([0, 1], labels)

    plt.yticks([0, 1], labels)

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "confusion_matrix.png")

    plt.close()


# ============================================================
# ROC curve
# ============================================================


def plot_roc_curve(
    y_true,
    y_probability,
):
    """
    Generate ROC curve.
    """

    fpr, tpr, _ = roc_curve(y_true, y_probability)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("ROC Curve")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "roc_curve.png")

    plt.close()


# ============================================================
# Feature importance
# ============================================================


def plot_feature_importance(
    model,
    feature_names,
    top_n=15,
):
    """
    Plot most important features
    from Random Forest.
    """

    classifier = model.named_steps["classifier"]

    importance = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))

    plt.barh(importance_df["feature"], importance_df["importance"])

    plt.gca().invert_yaxis()

    plt.title("Feature Importance")

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "feature_importance.png")

    plt.close()


# ============================================================
# Random Forest complexity curve
# ============================================================


def plot_tree_accuracy_curve(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessing_pipeline,
    random_state=42,
):
    """
    Plot accuracy depending on
    number of trees.

    Equivalent of learning curve
    for Random Forest.
    """

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.pipeline import Pipeline

    tree_numbers = [
        10,
        25,
        50,
        100,
        150,
        200,
    ]

    train_scores = []

    test_scores = []

    for trees in tree_numbers:

        pipeline = Pipeline(
            [
                (
                    "preprocessing",
                    preprocessing_pipeline,
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=trees,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        train_prediction = pipeline.predict(X_train)

        test_prediction = pipeline.predict(X_test)

        train_scores.append(accuracy_score(y_train, train_prediction))

        test_scores.append(accuracy_score(y_test, test_prediction))

    plt.figure(figsize=(8, 5))

    plt.plot(tree_numbers, train_scores, marker="o", label="Train Accuracy")

    plt.plot(tree_numbers, test_scores, marker="o", label="Test Accuracy")

    plt.title("Random Forest Trees vs Accuracy")

    plt.xlabel("Number of Trees")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "trees_vs_accuracy.png")

    plt.close()
