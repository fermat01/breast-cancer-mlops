"""
Model evaluation module.

Responsibilities
----------------
- Evaluate trained machine learning models
- Calculate classification metrics
- Generate evaluation reports
- Call visualization functions


This module does NOT:
- Load datasets
- Split datasets
- Train models
- Create plots directly
- Track MLflow experiments
"""

from pathlib import Path


import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from training.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_tree_accuracy_curve,
)

# ============================================================
# Directories
# ============================================================


BASE_DIR = Path(".")


EVALUATION_DIR = BASE_DIR / "reports" / "evaluation"


EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Metrics
# ============================================================


def calculate_metrics(
    y_true,
    y_pred,
    y_probability=None,
):
    """
    Calculate classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_probability is not None:

        metrics["roc_auc"] = roc_auc_score(y_true, y_probability)

    return metrics


# ============================================================
# Reports
# ============================================================


def create_classification_report(
    y_true,
    y_pred,
):
    """
    Generate classification report dataframe.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )

    return pd.DataFrame(report).transpose()


def create_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Generate confusion matrix dataframe.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    return pd.DataFrame(
        matrix,
        columns=[
            "Predicted_0",
            "Predicted_1",
        ],
        index=[
            "Actual_0",
            "Actual_1",
        ],
    )


# ============================================================
# Main evaluation function
# ============================================================


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Complete model evaluation pipeline.

    Parameters
    ----------
    model:
        Trained sklearn pipeline.

    X_train:
        Training features.

    X_test:
        Testing features.

    y_train:
        Training labels.

    y_test:
        Testing labels.
    """

    # ----------------------------------------
    # Predictions
    # ----------------------------------------

    y_pred = model.predict(X_test)

    y_probability = None

    if hasattr(model, "predict_proba"):

        y_probability = model.predict_proba(X_test)[:, 1]

    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability,
    )

    pd.DataFrame([metrics]).to_csv(
        EVALUATION_DIR / "metrics.csv",
        index=False,
    )

    # ----------------------------------------
    # Classification report
    # ----------------------------------------

    report_df = create_classification_report(
        y_test,
        y_pred,
    )

    report_df.to_csv(EVALUATION_DIR / "classification_report.csv")

    # ----------------------------------------
    # Confusion matrix
    # ----------------------------------------

    confusion_df = create_confusion_matrix(
        y_test,
        y_pred,
    )

    confusion_df.to_csv(EVALUATION_DIR / "confusion_matrix.csv")

    # ----------------------------------------
    # Visualizations
    # ----------------------------------------

    plot_confusion_matrix(
        y_test,
        y_pred,
    )

    if y_probability is not None:

        plot_roc_curve(
            y_test,
            y_probability,
        )

    # Random Forest feature importance

    if "classifier" in model.named_steps:
        plot_feature_importance(
            model,
            X_test.columns,
        )

    plot_tree_accuracy_curve(
        X_train,
        X_test,
        y_train,
        y_test,
        model.named_steps["preprocessing"],
    )

    return metrics
