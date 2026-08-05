"""
Training pipeline entry point.

Current responsibility:
- Orchestrate data loading
- Validate dataset
- Split dataset
- Apply preprocessing

Future responsibility:
- Train model
- Evaluate model
- Log experiments with MLflow
- Save model artifact
"""


from training.data_loader import load_dataset
from training.validate import validate_dataset
from training.split import split_dataset
from training.preprocess import (
    create_preprocessing_pipeline,
    preprocess_features,
)


def run_training_pipeline():
    """
    Execute the complete data preparation pipeline.

    Returns
    -------
    tuple
        Processed training and testing data.
    """

    print("=" * 60)
    print("Starting Training Pipeline")
    print("=" * 60)


    # --------------------------------
    # Step 1: Load dataset
    # --------------------------------

    print("\n[1/4] Loading dataset...")

    dataset = load_dataset()

    print(
        f"Dataset loaded: {dataset.features.shape}"
    )


    # --------------------------------
    # Step 2: Validate dataset
    # --------------------------------

    print("\n[2/4] Validating dataset...")

    validation_result = validate_dataset(
        dataset
    )


    if not validation_result.is_valid:

        print("\nValidation failed:")

        for error in validation_result.errors:
            print(
                f"- {error}"
            )

        raise ValueError(
            "Dataset validation failed"
        )


    print("Dataset validation successful")


    if validation_result.warnings:

        print("\nWarnings:")

        for warning in validation_result.warnings:
            print(
                f"- {warning}"
            )


    # --------------------------------
    # Step 3: Split dataset
    # --------------------------------

    print("\n[3/4] Splitting dataset...")

    split = split_dataset(
        dataset
    )

    print(
        f"Training samples: {len(split.X_train)}"
    )

    print(
        f"Testing samples: {len(split.X_test)}"
    )


    # --------------------------------
    # Step 4: Preprocessing
    # --------------------------------

    print("\n[4/4] Applying preprocessing...")


    preprocessing_pipeline = (
        create_preprocessing_pipeline()
    )


    X_train_processed, X_test_processed = (
        preprocess_features(
            preprocessing_pipeline,
            split.X_train,
            split.X_test,
        )
    )


    print(
        "Preprocessing completed"
    )

    print(
        "Processed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed testing shape:",
        X_test_processed.shape
    )


    print("\nPipeline completed successfully")


    return (
        X_train_processed,
        X_test_processed,
        split.y_train,
        split.y_test,
        preprocessing_pipeline,
    )


if __name__ == "__main__":

    run_training_pipeline()