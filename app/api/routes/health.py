"""
Health and readiness endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from app.core.config import get_settings
from app.schemas.health import (
    HealthResponse,
    ReadinessResponse,
)
from app.services.model_loader import is_model_loaded

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=HealthResponse,
)
def health_check():
    """
    Basic application health check.
    """

    settings = get_settings()

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
)
def readiness_check():
    """
    Check whether the API is ready to serve predictions.
    """

    model_loaded = is_model_loaded()

    if not model_loaded:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded.",
        )

    return ReadinessResponse(
        status="ready",
        model_loaded=True,
    )