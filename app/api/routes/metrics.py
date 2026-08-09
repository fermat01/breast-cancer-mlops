"""
Prometheus metrics endpoint.
"""

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"],
)


@router.get(
    "",
    include_in_schema=False,
)
def metrics():
    """
    Return Prometheus metrics.
    """

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )