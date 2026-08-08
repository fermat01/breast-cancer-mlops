"""
Model information endpoints.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.services.model_loader import is_model_loaded

router = APIRouter(
    prefix="/model",
    tags=["Model"],
)


@router.get("")
def model_info():
    """
    Return information about the currently configured model.
    """

    settings = get_settings()

    return {
        "name": settings.model_name,
        "alias": settings.model_alias,
        "uri": settings.model_uri,
        "loaded": is_model_loaded(),
    }