"""
Application Prometheus metrics.

Defines metrics used to monitor the FastAPI
prediction service.
"""

from prometheus_client import Counter, Gauge, Histogram


# ============================================================
# Prediction metrics
# ============================================================

PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total number of prediction requests.",
)

PREDICTION_SUCCESS_TOTAL = Counter(
    "prediction_success_total",
    "Total number of successful predictions.",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests.",
)


# ============================================================
# Prediction latency
# ============================================================

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Time spent generating predictions.",
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)


# ============================================================
# Prediction classes
# ============================================================

PREDICTIONS_BY_CLASS = Counter(
    "predictions_by_class_total",
    "Total number of predictions by predicted class.",
    ["class_label"],
)


# ============================================================
# Model information
# ============================================================

MODEL_INFO = Gauge(
    "model_info",
    "Information about the currently loaded ML model.",
    ["model_name", "model_alias", "model_version"],
)