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

from fastapi import FastAPI

from app.api.routes import health
from app.api.routes import metrics
from app.api.routes import model
from app.api.routes import prediction
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
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
    description=("Breast Cancer Classification API " "powered by MLflow."),
    lifespan=lifespan,
)


# ============================================================
# API routes
# ============================================================

app.include_router(
    health.router,
    prefix=settings.api_prefix,
)

app.include_router(
    prediction.router,
    prefix=settings.api_prefix,
)

app.include_router(
    model.router,
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
