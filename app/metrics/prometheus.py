"""
Prometheus metrics definitions.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
)

# Total API requests

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    [
        "endpoint",
        "method",
        "status",
    ],
)


# Request latency

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    [
        "endpoint",
    ],
)


# Predictions counter

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Prediction distribution",
    [
        "class_name",
    ],
)


# Model loaded status

MODEL_STATUS = Gauge(
    "model_loaded_status",
    "Model loading status",
)
