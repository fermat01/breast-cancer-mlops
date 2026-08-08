"""
Model information endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.services.model_loader import (
    get_model_metadata,
    is_model_loaded,
)

router = APIRouter(
    prefix="/model",
    tags=["Model"],
)


@router.get("")
def model_info():
    """
    Return information about the currently loaded MLflow model.
    """

    settings = get_settings()

    # --------------------------------------------------------
    # Check whether model is loaded
    # --------------------------------------------------------

    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded.",
        )

    # --------------------------------------------------------
    # Get loaded model metadata
    # --------------------------------------------------------

    try:
        metadata = get_model_metadata()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Return model information
    # --------------------------------------------------------

    return {
        "name": metadata.name,
        "alias": metadata.alias,
        "version": metadata.version,
        "run_id": metadata.run_id,
        "source": metadata.source,
        "uri": settings.model_uri,
        "loaded": True,
    }