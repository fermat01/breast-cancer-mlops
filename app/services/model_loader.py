"""
MLflow model loading service.
"""
from app.metrics.prometheus import MODEL_STATUS
import mlflow
import mlflow.pyfunc

from app.core.config import (
    MODEL_URI,
    MLFLOW_TRACKING_URI,
)

_model = None


def load_model():
    """
    Load MLflow champion model.

    Model is loaded once
    when API starts.
    """

    global _model

    if _model is None:

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        _model = mlflow.pyfunc.load_model(MODEL_URI)

        MODEL_STATUS.set(1)


    return _model
