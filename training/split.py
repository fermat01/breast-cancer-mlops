"""
Dataset splitting module.

Responsibilities:
- Split dataset into training and testing sets
- Maintain class distribution
- Ensure reproducibility
"""

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from training.data_loader import Dataset


@dataclass(frozen=True)
class SplitDataset:
    X_train: object
    X_test: object
    y_train: object
    y_test: object


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitDataset:

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )

    return SplitDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":

    from training.data_loader import load_dataset

    dataset = load_dataset()

    split = split_dataset(dataset)

    print("=" * 60)
    print("Dataset Split")
    print("=" * 60)

    print(f"Training samples: {len(split.X_train)}")
    print(f"Testing samples : {len(split.X_test)}")

    print("\nTraining distribution:")
    print(split.y_train.value_counts(normalize=True))

    print("\nTesting distribution:")
    print(split.y_test.value_counts(normalize=True))