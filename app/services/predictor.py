"""
Prediction service.

Responsible for:

- Validating feature count
- Building the pandas DataFrame
- Calling the loaded MLflow model
- Extracting prediction probabilities
- Mapping probabilities to class labels
- Returning a normalized prediction result
"""

from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.features import CLASS_LABELS, FEATURE_NAMES
from app.core.logging import get_logger
from app.services.model_loader import (
    get_model,
    get_model_metadata,
)

logger = get_logger(__name__)


def predict(features: list[float]) -> dict[str, Any]:
    """
    Generate a prediction using the loaded MLflow model.

    Parameters
    ----------
    features:
        Exactly 30 numerical features in the order expected
        by the trained model.

    Returns
    -------
    dict
        Prediction result containing:

        - prediction
        - prediction_label
        - probabilities
        - model_name
        - model_alias

    Raises
    ------
    ValueError
        If the number of features does not match the model schema.

    RuntimeError
        If the ML model has not been loaded.
    """

    settings = get_settings()

    # ========================================================
    # Validate feature count
    # ========================================================

    expected_features = len(FEATURE_NAMES)
    received_features = len(features)

    if received_features != expected_features:
        raise ValueError(
            f"Invalid number of features. "
            f"Expected {expected_features}, "
            f"received {received_features}."
        )

    # ========================================================
    # Get loaded model and metadata
    # ========================================================

    model = get_model()
    model_metadata = get_model_metadata()
    logger.debug( "Using model metadata: name=%s alias=%s version=%s run_id=%s", model_metadata.name, model_metadata.alias, model_metadata.version, model_metadata.run_id, )
    # ========================================================
    # Build DataFrame
    # ========================================================

    dataframe = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    logger.debug(
        "Prediction input shape: %s",
        dataframe.shape,
    )

    logger.debug(
        "Prediction features: %s",
        dataframe.columns.tolist(),
    )

    # ========================================================
    # Generate prediction
    # ========================================================

    logger.info(
        "Running prediction using model=%s alias=%s",
        settings.model_name,
        settings.model_alias,
    )

    prediction = model.predict(dataframe)

    predicted_class = int(prediction[0])

    # ========================================================
    # Map prediction to human-readable label
    # ========================================================

    prediction_label = CLASS_LABELS.get(
        predicted_class,
        str(predicted_class),
    )

    logger.info(
        "Prediction result: class=%s label=%s",
        predicted_class,
        prediction_label,
    )

    # ========================================================
    # Generate probabilities
    # ========================================================

    probabilities: dict[str, float] | None = None

    try:
        underlying_model = model._model_impl.sklearn_model

        if hasattr(underlying_model, "predict_proba"):
            proba = underlying_model.predict_proba(dataframe)[0]

            classes = underlying_model.classes_

            probabilities = {
                CLASS_LABELS.get(
                    int(class_label),
                    str(class_label),
                ): float(probability)
                for class_label, probability in zip(
                    classes,
                    proba,
                )
            }

            logger.debug(
                "Prediction probabilities: %s",
                probabilities,
            )

        else:
            logger.warning(
                "Loaded model does not expose predict_proba()."
            )

    except AttributeError as exc:
        logger.warning(
            "Unable to access underlying sklearn model: %s",
            exc,
        )

    except Exception:
        logger.exception(
            "Unexpected error while calculating prediction probabilities."
        )

    # ========================================================
    # Return normalized result
    # ========================================================

    return {
        "prediction": predicted_class,
        "prediction_label": prediction_label,
        "probabilities": probabilities,
        "model_name": settings.model_name,
        "model_alias": settings.model_alias,
       "model_version": model_metadata.version,
        "model_run_id": model_metadata.run_id,
    }