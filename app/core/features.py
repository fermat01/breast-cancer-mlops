"""
Breast cancer model feature definitions.

These names MUST exactly match the column names
used during model training.
"""

FEATURE_NAMES = [
    # Mean features
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    # Standard error features
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    # Worst features
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

# ============================================================
# Output classes
# ============================================================

CLASS_LABELS: dict[int, str] = {
    0: "malignant",
    1: "benign",
}

N_FEATURES = len(FEATURE_NAMES)