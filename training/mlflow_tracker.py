"""
MLflow tracking utilities.

Responsibilities
-----------------
- Configure MLflow tracking
- Create experiments
- Start MLflow runs
- Log parameters
- Log metrics
- Log artifacts
- Register models
- Manage model aliases

Compatible with:
- Local development
- Docker
- MLflow Model Registry
"""

from pathlib import Path
import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ============================================================
# Configuration
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)


EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "breast-cancer-classification",
)


# Local development artifact folder
ARTIFACTS_DIR = Path(
    os.getenv(
        "MLFLOW_ARTIFACTS_DIR",
        str(PROJECT_ROOT / "mlartifacts"),
    )
)


ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ARTIFACT_LOCATION = ARTIFACTS_DIR.resolve().as_uri()

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "breast-cancer-classifier",
)

# ============================================================
# MLflow configuration
# ============================================================


def configure_mlflow():
    """
    Configure MLflow backend and experiment.
    """

    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:

        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACT_LOCATION,
        )

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
# Logging
# ============================================================


def log_parameters(
    params: dict,
):
    """
    Log ML parameters.
    """

    mlflow.log_params(params)


def log_metrics(
    metrics: dict,
):
    """
    Log evaluation metrics.
    """

    mlflow.log_metrics(metrics)


def log_directory(
    directory: Path,
):
    """
    Log directory as MLflow artifact.
    """

    if directory.exists():

        mlflow.log_artifacts(str(directory))


# ============================================================
# Model logging
# ============================================================


def log_model(
    model,
):
    """
    Log sklearn model and register it.
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
# Helper
# ============================================================


def get_model_uri(
    alias: str = "champion",
):
    """
    Return MLflow model URI.
    """

    return f"models:/{MODEL_NAME}@{alias}"
