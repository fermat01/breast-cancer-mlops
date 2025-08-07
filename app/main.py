from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import time
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram
import joblib
from utils import preprocess_features, map_prediction_to_label

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Request count", ["endpoint", "http_method", "http_status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

# Define the Pydantic model with all 30 features
class BreastCancerFeatures(BaseModel):
    mean_radius: float = Field(...)
    mean_texture: float = Field(...)
    mean_perimeter: float = Field(...)
    mean_area: float = Field(...)
    mean_smoothness: float = Field(...)
    mean_compactness: float = Field(...)
    mean_concavity: float = Field(...)
    mean_concave_points: float = Field(...)
    mean_symmetry: float = Field(...)
    mean_fractal_dimension: float = Field(...)
    radius_error: float = Field(...)
    texture_error: float = Field(...)
    perimeter_error: float = Field(...)
    area_error: float = Field(...)
    smoothness_error: float = Field(...)
    compactness_error: float = Field(...)
    concavity_error: float = Field(...)
    concave_points_error: float = Field(...)
    symmetry_error: float = Field(...)
    fractal_dimension_error: float = Field(...)
    worst_radius: float = Field(...)
    worst_texture: float = Field(...)
    worst_perimeter: float = Field(...)
    worst_area: float = Field(...)
    worst_smoothness: float = Field(...)
    worst_compactness: float = Field(...)
    worst_concavity: float = Field(...)
    worst_concave_points: float = Field(...)
    worst_symmetry: float = Field(...)
    worst_fractal_dimension: float = Field(...)

# Load model once when starting app
model = joblib.load("rf_model.joblib")

@app.on_event("startup")
async def startup_event():
    start_http_server(8001)  # Prometheus metrics server

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    REQUEST_COUNT.labels(endpoint=request.url.path, http_method=request.method, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    return response

@app.post("/predict")
def predict_endpoint(data: BreastCancerFeatures):
    try:
        # Convert input data to dict
        input_dict = data.dict()
        # Preprocess features to numpy array
        features = preprocess_features(input_dict)
        # Get numeric prediction
        pred = model.predict(features)[0]
        # Map to label
        label = map_prediction_to_label(pred)
        return {"prediction": label}
    except ValueError as e:
        # Missing or invalid input feature
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
