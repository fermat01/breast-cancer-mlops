"""
FastAPI application.

Responsibilities:

- Expose REST API endpoints
- Validate requests using Pydantic
- Load MLflow registered model
- Serve predictions
- Expose Prometheus metrics
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from app.api.v1.endpoints import metrics
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.metrics import INVALID_INPUTS_TOTAL
from app.services.model_loader import load_model

# ============================================================
# Application configuration
# ============================================================

configure_logging()

logger = get_logger(__name__)

settings = get_settings()


# ============================================================
# Application lifespan
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.
    """

    logger.info(
        "Starting %s v%s",
        settings.app_name,
        settings.app_version,
    )

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    logger.info("Loading ML model...")

    load_model()

    logger.info("Application startup completed.")

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    logger.info("Application shutting down.")


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Breast Cancer Classification API powered by MLflow.",
    lifespan=lifespan,
)


# ============================================================
# Request validation monitoring
# ============================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Record invalid prediction requests while preserving
    FastAPI's standard validation error response.
    """

    reason = "other"

    for error in exc.errors():
        location = error.get("loc", ())
        error_type = error.get("type", "")

        if "features" not in location:
            continue

        if error_type == "missing":
            reason = "missing_field"
            break

        if error_type in {
            "too_short",
            "too_long",
        }:
            reason = "feature_count"
            break

        if (
            "parsing" in error_type
            or "type" in error_type
            or error_type.startswith("float_")
        ):
            reason = "invalid_type"
            break

    if request.url.path == f"{settings.api_prefix}/predictions":
        INVALID_INPUTS_TOTAL.labels(reason=reason).inc()

        logger.warning(
            "Invalid prediction input: path=%s reason=%s",
            request.url.path,
            reason,
        )

    return await request_validation_exception_handler(
        request,
        exc,
    )


# ============================================================
# Versioned API routes
# ============================================================

app.include_router(
    api_router,
    prefix=settings.api_prefix,
)


# ============================================================
# Prometheus metrics
# ============================================================

# Keep Prometheus metrics outside /api/v1.
# Prometheus will scrape:
# http://fastapi:8000/metrics

app.include_router(
    metrics.router,
)


# ============================================================
# Root endpoint
# ============================================================


@app.get("/")
def root():
    """
    API root endpoint.
    """

    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "ready": f"{settings.api_prefix}/health/ready",
    }
