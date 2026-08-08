"""
MLflow model loading service.

Responsibilities:

- Load the MLflow model during application startup
- Keep the loaded model in memory
- Resolve the configured MLflow model alias
- Expose model metadata/version
- Provide model loading status
"""

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# Model metadata
# ============================================================


@dataclass(frozen=True)
class ModelMetadata:
    """
    Metadata for the currently loaded MLflow model.
    """

    name: str
    alias: str
    version: str
    run_id: str | None = None
    source: str | None = None


# ============================================================
# Global model state
# ============================================================

_model: Any | None = None

_model_metadata: ModelMetadata | None = None


# ============================================================
# Load model
# ============================================================


def load_model() -> Any:
    """
    Load the MLflow model configured by the application.

    The model is loaded once during application startup.
    """

    global _model
    global _model_metadata

    settings = get_settings()

    model_name = settings.model_name
    model_alias = settings.model_alias
    tracking_uri = settings.mlflow_tracking_uri

    logger.info(
        "Loading MLflow model: %s",
        settings.model_uri,
    )

    logger.info(
        "MLflow tracking URI: %s",
        tracking_uri,
    )

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(
        tracking_uri=tracking_uri,
    )

    # --------------------------------------------------------
    # Resolve model alias
    # --------------------------------------------------------

    try:
        model_version = client.get_model_version_by_alias(
            name=model_name,
            alias=model_alias,
        )

    except Exception as exc:
        logger.exception(
            "Unable to resolve MLflow model alias: " "model=%s alias=%s",
            model_name,
            model_alias,
        )

        raise RuntimeError(
            f"Unable to resolve MLflow model " f"'{model_name}@{model_alias}'."
        ) from exc

    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    version = str(model_version.version)

    logger.info(
        "Resolved MLflow model: " "model=%s alias=%s version=%s",
        model_name,
        model_alias,
        version,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:
        loaded_model = mlflow.pyfunc.load_model(
            settings.model_uri,
        )

    except Exception as exc:
        logger.exception(
            "Failed to load MLflow model: %s",
            settings.model_uri,
        )

        raise RuntimeError(
            f"Unable to load MLflow model " f"'{settings.model_uri}'."
        ) from exc

    # --------------------------------------------------------
    # Store model
    # --------------------------------------------------------

    _model = loaded_model

    # --------------------------------------------------------
    # Store metadata
    # --------------------------------------------------------

    _model_metadata = ModelMetadata(
        name=model_name,
        alias=model_alias,
        version=version,
        run_id=model_version.run_id,
        source=model_version.source,
    )

    logger.info(
        "MLflow model loaded successfully: " "model=%s alias=%s version=%s run_id=%s",
        model_name,
        model_alias,
        version,
        model_version.run_id,
    )

    return _model


# ============================================================
# Get model
# ============================================================


def get_model() -> Any:
    """
    Return the currently loaded MLflow model.

    Raises
    ------
    RuntimeError
        If the model has not been loaded.
    """

    if _model is None:
        raise RuntimeError("ML model has not been loaded.")

    return _model


# ============================================================
# Check model status
# ============================================================


def is_model_loaded() -> bool:
    """
    Return True if the MLflow model is loaded.
    """

    return _model is not None


# ============================================================
# Get model metadata
# ============================================================


def get_model_metadata() -> ModelMetadata:
    """
    Return metadata for the currently loaded model.

    Raises
    ------
    RuntimeError
        If model metadata is not available.
    """

    if _model_metadata is None:
        raise RuntimeError("ML model metadata is not available.")

    return _model_metadata
