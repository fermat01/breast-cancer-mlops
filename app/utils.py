import numpy as np

FEATURE_ORDER = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error", "smoothness_error",
    "compactness_error", "concavity_error", "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
    "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
]

def preprocess_features(input_data: dict) -> np.ndarray:
    """
    Converts input dictionary of features to a numpy array ordered according to FEATURE_ORDER.
    Raises ValueError if any feature is missing.
    """
    try:
        feature_vector = np.array([[input_data[feat] for feat in FEATURE_ORDER]])
        return feature_vector
    except KeyError as e:
        missing = e.args[0]
        raise ValueError(f"Missing feature in input data: {missing}")

def map_prediction_to_label(prediction: int) -> str:
    """
    Maps model numeric prediction to label string.
    According to sklearn breast cancer dataset: 0=malignant, 1=benign
    """
    mapping = {0: "malignant", 1: "benign"}
    return mapping.get(prediction, "unknown")
