from fastapi import APIRouter

from app.api.v1.endpoints import health, metrics, model, prediction

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(metrics.router)
api_router.include_router(model.router)
api_router.include_router(prediction.router)
