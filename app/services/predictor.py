"""
Prediction service.
"""

import pandas as pd

from app.services.model_loader import load_model

from app.metrics.prometheus import (
    PREDICTION_COUNT,
)


def predict(features: dict):

    model = load_model()

    dataframe = pd.DataFrame([features])

    prediction = model.predict(dataframe)

    result = int(prediction[0])

    label = "malignant" if result == 0 else "benign"

    PREDICTION_COUNT.labels(class_name=label).inc()

    return result
