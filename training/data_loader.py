"""
Data loading module.

This module is responsible for loading the Breast Cancer Wisconsin
dataset and returning it as pandas DataFrames.

Responsibilities:
- Load dataset
- Return features and target
- Provide metadata

It should NOT:
- Validate data
- Preprocess data
- Split data
- Train models
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass(frozen=True)
class Dataset:
    """
    Container for the loaded dataset.
    """

    features: pd.DataFrame
    target: pd.Series
    target_names: list[str]
    feature_names: list[str]
    description: str


def load_dataset() -> Dataset:
    """
    Load the Breast Cancer Wisconsin dataset.

    Returns
    -------
    Dataset
        Dataset object containing features, target,
        metadata, and description.
    """

    dataset = load_breast_cancer()

    features = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names,
    )

    target = pd.Series(
        dataset.target,
        name="target",
    )

    return Dataset(
        features=features,
        target=target,
        target_names=list(dataset.target_names),
        feature_names=list(dataset.feature_names),
        description=dataset.DESCR,
    )


if __name__ == "__main__":
    data = load_dataset()

    print("=" * 60)
    print("Breast Cancer Dataset")
    print("=" * 60)

    print(f"Samples : {len(data.features)}")
    print(f"Features: {len(data.feature_names)}")

    print("\nFeature names:")
    print(data.feature_names)

    print("\nTarget names:")
    print(data.target_names)

    print("\nFirst five rows:")
    print(data.features.head())

    print("\nTarget distribution:")
    print(data.target.value_counts())