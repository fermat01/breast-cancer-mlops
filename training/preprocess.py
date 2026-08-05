"""
Data preprocessing module.

Responsibilities:
- Build preprocessing pipeline
- Transform raw features into ML-ready features

This module should NOT:
- Load datasets
- Split datasets
- Train models
"""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_preprocessing_pipeline() -> Pipeline:
    """
    Create the preprocessing pipeline.

    Returns
    -------
    Pipeline
        Scikit-learn preprocessing pipeline.

    Notes
    -----
    The Breast Cancer Wisconsin dataset contains
    numerical features with different scales.

    Example:

    mean area:
        ~600

    mean smoothness:
        ~0.1

    Scaling prevents features with larger values
    from dominating the model.
    """

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    return pipeline


def preprocess_features(
    pipeline: Pipeline,
    X_train,
    X_test,
):
    """
    Fit preprocessing pipeline on training data
    and transform train/test datasets.

    Parameters
    ----------
    pipeline:
        sklearn preprocessing pipeline.

    X_train:
        Training features.

    X_test:
        Testing features.

    Returns
    -------
    tuple
        Transformed training and testing data.
    """

    X_train_processed = pipeline.fit_transform(
        X_train
    )

    X_test_processed = pipeline.transform(
        X_test
    )

    return (
        X_train_processed,
        X_test_processed,
    )


if __name__ == "__main__":

    from training.data_loader import load_dataset
    from training.validate import validate_dataset
    from training.split import split_dataset


    dataset = load_dataset()

    validation = validate_dataset(dataset)

    if not validation.is_valid:
        raise ValueError(
            validation.errors
        )

    X_train, X_test, y_train, y_test = split_dataset(
        dataset
    )

    preprocessing = create_preprocessing_pipeline()

    X_train_processed, X_test_processed = preprocess_features(
        preprocessing,
        X_train,
        X_test,
    )

    print("=" * 60)
    print("Preprocessing Test")
    print("=" * 60)

    print(
        "Original shape:",
        X_train.shape
    )

    print("Processed shape:",
       X_train_processed.shape
    )