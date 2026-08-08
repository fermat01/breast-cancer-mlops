"""
FastAPI dependencies.
"""

from app.services.model_loader import get_model


def get_prediction_model():
    """
    Dependency that provides the loaded ML model.
    """

    return get_model()