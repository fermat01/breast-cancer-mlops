"""
API request and response schemas.
"""

from pydantic import BaseModel, Field


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
