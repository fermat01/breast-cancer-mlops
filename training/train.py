"""
Main ML training pipeline.

Responsibilities
----------------

- Orchestrate complete workflow


Workflow
--------

Load data
    |
Validate
    |
Split
    |
Preprocess
    |
Train
    |
Evaluate
    |
MLflow Tracking
    |
Model Registry
"""

import logging
from pathlib import Path


from training.data_loader import load_dataset

from training.validate import validate_dataset

from training.split import split_dataset


from training.preprocess import (
    create_preprocessing_pipeline,
)


from training.model import (
    build_model,
    train_model,
)


from training.evaluate import (
    evaluate_model,
)


from training.mlflow_tracker import (
    start_run,
    log_parameters,
    log_metrics,
    log_directory,
    log_model,
)

# ============================================================
# Logging configuration
# ============================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


logger = logging.getLogger(__name__)


# ============================================================
# Training pipeline
# ============================================================


def run_training_pipeline():
    """
    Execute complete ML training pipeline.
    """

    logger.info("Starting training pipeline")

    # =====================================================
    # Start MLflow Run
    # =====================================================

    with start_run(run_name="RandomForest") as run:

        # =================================================
        # MLflow Run ID
        # =================================================

        run_id = run.info.run_id

        logger.info(f"MLflow Run ID: {run_id}")

        # =================================================
        # 1. Load dataset
        # =================================================

        logger.info("Loading dataset")

        dataset = load_dataset()

        # =================================================
        # 2. Validate dataset
        # =================================================

        logger.info("Validating dataset")

        validation = validate_dataset(dataset)

        if not validation.is_valid:

            raise ValueError(validation.errors)

        logger.info("Dataset validation successful")

        # =================================================
        # 3. Split dataset
        # =================================================

        logger.info("Splitting dataset")

        split = split_dataset(dataset)

        logger.info(f"Train shape : {split.X_train.shape}")

        logger.info(f"Test shape  : {split.X_test.shape}")

        # =================================================
        # 4. Preprocessing
        # =================================================

        logger.info("Creating preprocessing pipeline")

        preprocessing = create_preprocessing_pipeline()

        # =================================================
        # 5. Build model
        # =================================================

        logger.info("Building model")

        model = build_model(
            preprocessing_pipeline=preprocessing,
            n_estimators=100,
            random_state=42,
        )

        # =================================================
        # 6. Train model
        # =================================================

        logger.info("Training model")

        trained_model = train_model(
            model,
            split.X_train,
            split.y_train,
        )

        logger.info("Training completed")

        # =================================================
        # 7. Evaluate model
        # =================================================

        logger.info("Evaluating model")

        metrics = evaluate_model(
            trained_model,
            split.X_train,
            split.X_test,
            split.y_train,
            split.y_test,
        )

        logger.info(f"Metrics: {metrics}")

        # =================================================
        # 8. MLflow Tracking
        # =================================================

        logger.info("Logging experiment to MLflow")

        parameters = {
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42,
            "test_size": 0.20,
            "features": split.X_train.shape[1],
            "training_samples": len(split.X_train),
            "testing_samples": len(split.X_test),
        }

        log_parameters(parameters)

        log_metrics(metrics)

        # =================================================
        # 9. Log and Register model
        # =================================================

        logger.info("Logging trained model")

        log_model(trained_model)

        logger.info("Model registered successfully")

        # =================================================
        # 10. Log evaluation artifacts
        # =================================================

        reports_dir = Path("reports") / "evaluation"

        if reports_dir.exists():

            logger.info("Logging evaluation artifacts")

            log_directory(reports_dir)

        logger.info("MLflow logging completed")

        return (
            trained_model,
            metrics,
        )


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":

    model, metrics = run_training_pipeline()

    print("\n")

    print("=" * 60)

    print("Training completed successfully")

    print("=" * 60)

    for metric_name, metric_value in metrics.items():

        print(f"{metric_name:<15}: {metric_value:.4f}")
