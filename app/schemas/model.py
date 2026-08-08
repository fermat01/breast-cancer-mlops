"""
Model API schemas.

Defines response models for MLflow model information.
"""

from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """
    Response containing metadata for the currently loaded MLflow model.
    """

    name: str = Field(
        ...,
        description="Registered MLflow model name.",
        examples=["breast-cancer-classifier"],
    )

    alias: str = Field(
        ...,
        description="MLflow model alias currently being served.",
        examples=["champion"],
    )

    version: str = Field(
        ...,
        description="MLflow model version currently being served.",
        examples=["1"],
    )

    run_id: str | None = Field(
        default=None,
        description="MLflow run ID associated with the model version.",
        examples=["a1b2c3d4e5f6"],
    )

    source: str | None = Field(
        default=None,
        description="MLflow source URI for the model version.",
    )