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

    Handles expected prediction failures and returns
    appropriate HTTP status codes.
    """

    try:
        result = predict(
            request.features
        )

        return PredictionResponse(
            **result
        )

    # --------------------------------------------------------
    # Invalid prediction input
    # --------------------------------------------------------

    except ValueError as exc:
        logger.warning(
            "Invalid prediction input: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Model/service unavailable
    # --------------------------------------------------------

    except RuntimeError as exc:
        logger.error(
            "Prediction service unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is currently unavailable.",
        ) from exc

    # --------------------------------------------------------
    # Unexpected prediction error
    # --------------------------------------------------------

    except Exception as exc:
        logger.exception(
            "Unexpected prediction error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the prediction.",
        ) from exc

