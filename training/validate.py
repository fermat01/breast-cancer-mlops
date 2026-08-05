"""
Dataset validation module.

This module verifies that the dataset is suitable
for machine learning training.

Responsibilities:
- Check missing values
- Check duplicated rows
- Validate feature count
- Validate target values
- Generate validation report

It should NOT:
- Modify the dataset
- Preprocess features
- Train models
"""


from dataclasses import dataclass

import pandas as pd

from training.data_loader import Dataset


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of dataset validation.
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_dataset(
    dataset: Dataset,
) -> ValidationResult:
    """
    Validate Breast Cancer dataset.

    Parameters
    ----------
    dataset:
        Loaded dataset object.

    Returns
    -------
    ValidationResult
        Validation status with errors and warnings.
    """

    errors: list[str] = []
    warnings: list[str] = []

    X = dataset.features
    y = dataset.target

    # -----------------------------
    # Check empty dataset
    # -----------------------------
    if X.empty:
        errors.append("Feature dataset is empty.")

    if y.empty:
        errors.append("Target dataset is empty.")

    # -----------------------------
    # Check missing values
    # -----------------------------
    missing_values = X.isnull().sum().sum()

    if missing_values > 0:
        errors.append(
            f"Dataset contains {missing_values} missing values."
        )

    # -----------------------------
    # Check duplicated rows
    # -----------------------------
    duplicates = X.duplicated().sum()

    if duplicates > 0:
        warnings.append(
            f"Dataset contains {duplicates} duplicated rows."
        )

    # -----------------------------
    # Validate feature count
    # -----------------------------
    expected_features = 30

    if len(dataset.feature_names) != expected_features:
        errors.append(
            f"Expected {expected_features} features, "
            f"found {len(dataset.feature_names)}."
        )

    # -----------------------------
    # Validate target values
    # -----------------------------
    unique_targets = sorted(y.unique())

    expected_targets = [0, 1]

    if unique_targets != expected_targets:
        errors.append(
            f"Unexpected target values: {unique_targets}"
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


if __name__ == "__main__":

    from training.data_loader import load_dataset

    dataset = load_dataset()

    result = validate_dataset(dataset)

    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    print(
        f"Valid: {result.is_valid}"
    )

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"- {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
