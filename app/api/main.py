"""
FastAPI application.

Responsibilities
----------------
- Expose REST API endpoints
- Validate requests using Pydantic
- Load MLflow registered model
- Serve predictions
- Expose Prometheus metrics
"""

from contextlib import asynccontextmanager

import time

from fastapi import FastAPI, Request

from fastapi.responses import Response

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)


from app.api.schemas import (
    BreastCancerFeatures,
)


from app.services.predictor import (
    predict,
)


from app.services.model_loader import (
    load_model,
)


from app.metrics.prometheus import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
)

# ============================================================
# Lifespan management
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Startup:
        - Load MLflow champion model

    Shutdown:
        - Cleanup resources if needed
    """

    # ----------------------------
    # Startup
    # ----------------------------

    print("Loading MLflow model...")

    load_model()

    print("MLflow model loaded successfully")

    yield

    # ----------------------------
    # Shutdown
    # ----------------------------

    print("Application shutting down")


# ============================================================
# FastAPI application
# ============================================================


app = FastAPI(
    title="Breast Cancer ML API",
    version="1.0.0",
    description=("Machine Learning API " "using FastAPI, Pydantic and MLflow"),
    lifespan=lifespan,
)


# ============================================================
# Prometheus middleware
# ============================================================


@app.middleware("http")
async def prometheus_middleware(
    request: Request,
    call_next,
):
    """
    Collect API metrics.
    """

    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path,
    ).observe(latency)

    return response


# ============================================================
# Health endpoint
# ============================================================


@app.get("/health")
def health():

    return {"status": "ok"}


# ============================================================
# Prediction endpoint
# ============================================================


@app.post("/predict")
def predict_endpoint(data: BreastCancerFeatures):

    prediction = predict(data.model_dump())

    label = "malignant" if prediction == 0 else "benign"

    return {
        "prediction": prediction,
        "label": label,
    }


# ============================================================
# Prometheus metrics endpoint
# ============================================================


@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
