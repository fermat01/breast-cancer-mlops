"""
MLflow tracking utilities.

Responsibilities
----------------
- Configure MLflow tracking
- Create experiments
- Start MLflow runs
- Log parameters
- Log metrics
- Log artifacts
- Register models
- Manage model aliases

Architecture
------------
Backend store:
    PostgreSQL via MLflow Server

Artifact store:
    MinIO S3 via MLflow Server

Compatible with:
    - Local development
    - Docker Compose
    - MLflow Model Registry
"""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

# ============================================================
# Configuration
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)


EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "breast-cancer-classification",
)


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "breast-cancer-classifier",
)


# ============================================================
# MLflow configuration
# ============================================================


def configure_mlflow():
    """
    Configure MLflow tracking server.

    MLflow server handles:
    - PostgreSQL backend
    - MinIO artifact storage
    - Model Registry
    """

    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:

        mlflow.create_experiment(name=EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Run management
# ============================================================


def start_run(
    run_name: str = "training-run",
):
    """
    Start MLflow run.
    """

    configure_mlflow()

    return mlflow.start_run(run_name=run_name)


# ============================================================
# Parameters logging
# ============================================================


def log_parameters(
    params: dict,
):
    """
    Log ML parameters.
    """

    mlflow.log_params(params)


# ============================================================
# Metrics logging
# ============================================================


def log_metrics(
    metrics: dict,
):
    """
    Log evaluation metrics.
    """

    mlflow.log_metrics(metrics)


# ============================================================
# Artifact logging
# ============================================================


def log_directory(
    directory: Path,
):
    """
    Upload directory artifacts.

    Example:
        reports/
        plots/
        evaluation files

    Stored automatically in MinIO.
    """

    if directory.exists():

        mlflow.log_artifacts(str(directory))


# ============================================================
# Model logging + Registry
# ============================================================


def log_model(
    model,
):
    """
    Log sklearn model.

    Registers model into MLflow Model Registry.
    """

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME,
    )

    return model_info


# ============================================================
# Model alias management
# ============================================================


def set_model_alias(
    version: int,
    alias: str = "champion",
):
    """
    Assign alias to registered model.

    Example:
        champion
    """

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=str(version),
    )

    return {
        "model": MODEL_NAME,
        "alias": alias,
        "version": version,
    }


# ============================================================
# Model URI helper
# ============================================================


def get_model_uri(
    alias: str = "champion",
):
    """
    Return MLflow Model Registry URI.

    Example:
        models:/breast-cancer-classifier@champion
    """

    return f"models:/{MODEL_NAME}@{alias}"
