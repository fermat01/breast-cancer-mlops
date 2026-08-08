"""
Prediction API schemas.

Defines request and response models for breast cancer predictions.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.core.features import CLASS_LABELS, N_FEATURES


class PredictionRequest(BaseModel):
    """
    Request body for a breast cancer prediction.

    Features must be provided in the exact order expected
    by the trained ML model.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "features": [
                        17.99,
                        10.38,
                        122.8,
                        1001.0,
                        0.1184,
                        0.2776,
                        0.3001,
                        0.1471,
                        0.2419,
                        0.07871,
                        1.095,
                        0.9053,
                        8.589,
                        153.4,
                        0.006399,
                        0.04904,
                        0.05373,
                        0.01587,
                        0.03003,
                        0.006193,
                        25.38,
                        17.33,
                        184.6,
                        2019.0,
                        0.1622,
                        0.6656,
                        0.7119,
                        0.2654,
                        0.4601,
                        0.1189,
                    ]
                }
            ]
        }
    )

    features: list[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=(
            f"Exactly {N_FEATURES} numerical features in the same "
            "order used during model training."
        ),
    )


class PredictionResponse(BaseModel):
    """
    Response returned after a successful prediction.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prediction": 0,
                    "prediction_label": "malignant",
                    "probabilities": {
                        "malignant": 0.97,
                        "benign": 0.03,
                    },
                    "model_name": "breast-cancer-classifier",
                    "model_alias": "champion",
                    "model_version": "1",
                    "model_run_id": "abc123...",
                }
            ]
        }
    )

    prediction: int = Field(
        ...,
        description=(f"Predicted class. Supported classes: {CLASS_LABELS}."),
        examples=[0],
    )

    prediction_label: str = Field(
        ...,
        description=(
            "Human-readable prediction label. "
            "Currently either 'malignant' or 'benign'."
        ),
        examples=["malignant"],
    )

    probabilities: dict[str, float] | None = Field(
        default=None,
        description=(
            "Prediction probability for each class. "
            "Values are between 0 and 1 and normally sum to 1."
        ),
        examples=[
            {
                "malignant": 0.97,
                "benign": 0.03,
            }
        ],
    )

    model_name: str = Field(
        ...,
        description="Registered MLflow model name.",
        examples=["breast-cancer-classifier"],
    )

    model_alias: str = Field(
        ...,
        description="MLflow model alias used for prediction.",
        examples=["champion"],
    )

    model_version: str = Field(
        ...,
        description="MLflow model version used for the prediction.",
        examples=["1"],
    )

    model_run_id: str | None = Field(
        default=None,
        description="MLflow run ID associated with the model version.",
        examples=["abc123..."],
    )
