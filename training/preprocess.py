"""
Data preprocessing module.

Responsibilities
----------------
- Create reusable preprocessing pipelines.
- Validate preprocessing configuration.

This module intentionally DOES NOT:
- Load datasets.
- Split datasets.
- Fit transformers.
- Transform datasets.
- Train machine learning models.

The preprocessing pipeline returned here is designed to be
embedded inside the final scikit-learn Pipeline together with
the classifier.

Example
-------

Pipeline(
    [
        ("preprocessing", create_preprocessing_pipeline()),
        ("classifier", RandomForestClassifier())
    ]
)

This guarantees that the exact same preprocessing is applied
during training and inference.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_preprocessing_pipeline() -> Pipeline:
    """
    Create the preprocessing pipeline.

    Returns
    -------
    Pipeline
        A scikit-learn preprocessing pipeline.

    Notes
    -----
    The Wisconsin Breast Cancer dataset contains only numerical
    features with different scales.

    Example
    -------
    mean area            ≈ 650
    mean smoothness      ≈ 0.09

    Although RandomForest does not strictly require scaling,
    including preprocessing inside the ML Pipeline ensures a
    consistent workflow and makes it easy to replace the model
    later (e.g. LogisticRegression, SVM, Neural Networks).
    """

    preprocessing_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            )
        ]
    )

    return preprocessing_pipeline


def pipeline_summary(pipeline: Pipeline) -> None:
    """
    Print information about the preprocessing pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Preprocessing pipeline.
    """

    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)

    print(f"Pipeline type : {type(pipeline).__name__}")
    print(f"Number of steps : {len(pipeline.steps)}")

    print("\nSteps")

    for index, (name, transformer) in enumerate(
        pipeline.steps,
        start=1,
    ):
        print(f"{index}. {name:<15} -> {transformer.__class__.__name__}")


def validate_pipeline(pipeline: Pipeline) -> bool:
    """
    Validate preprocessing pipeline.

    Parameters
    ----------
    pipeline : Pipeline

    Returns
    -------
    bool
        True if valid.
    """

    if not isinstance(pipeline, Pipeline):
        raise TypeError("Expected a scikit-learn Pipeline.")

    if len(pipeline.steps) == 0:
        raise ValueError("Pipeline contains no preprocessing steps.")

    return True


if __name__ == "__main__":

    pipeline = create_preprocessing_pipeline()

    validate_pipeline(pipeline)

    pipeline_summary(pipeline)

    print("\nPipeline successfully created.")
