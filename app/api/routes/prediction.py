"""
Prediction API endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from app.services.predictor import predict

logger = get_logger(__name__)

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


@router.post(
    "",
    response_model=PredictionResponse,
)
def create_prediction(
    request: PredictionRequest,
):
    """
    Generate a breast cancer prediction.
    """

    try:

        result = predict(
            request.features
        )

        return PredictionResponse(
            **result
        )

    except RuntimeError as exc:

        logger.error(
            "Prediction service unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    except Exception as exc:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc