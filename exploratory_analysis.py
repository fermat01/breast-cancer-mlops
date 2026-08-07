"""
Exploratory Data Analysis (EDA) Module
======================================

This module performs exploratory data analysis for the
Breast Cancer Wisconsin dataset.

Responsibilities
----------------
✓ Dataset overview
✓ Missing value analysis
✓ Duplicate analysis
✓ Descriptive statistics
✓ Advanced statistics
✓ Correlation analysis
✓ Outlier detection
✓ PCA analysis
✓ Report generation
✓ CSV export
✓ Markdown report

This module DOES NOT:
---------------------
- Train models
- Evaluate models
- Save ML models

Author:
Breast Cancer MLOps Project
"""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

from training.data_loader import Dataset

# =============================================================================
# Configuration
# =============================================================================

REPORTS_DIR = Path("reports")
STATISTICS_DIR = REPORTS_DIR / "statistics"

REPORTS_DIR.mkdir(exist_ok=True)
STATISTICS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("EDA")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

# =============================================================================
# Helper Functions
# =============================================================================


def print_section(title: str) -> None:
    """
    Print a formatted section title.
    """

    print("\n")
    print("=" * 80)
    print(title.upper())
    print("=" * 80)


def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    """
    Save DataFrame into reports/statistics.

    Parameters
    ----------
    dataframe
        DataFrame to save.

    filename
        CSV filename.

    Returns
    -------
    Path
        Saved file path.
    """

    output_path = STATISTICS_DIR / filename

    dataframe.to_csv(
        output_path,
        index=True,
    )

    logger.info("Saved CSV -> %s", output_path)

    return output_path


# =============================================================================
# Dataset Overview
# =============================================================================


def dataset_overview(dataset: Dataset) -> dict:
    """
    Display a high-level overview of the dataset.

    Parameters
    ----------
    dataset
        Loaded dataset.

    Returns
    -------
    dict
        Dataset metadata.
    """

    X = dataset.features
    y = dataset.target

    print_section("Dataset Overview")

    overview = {
        "Samples": X.shape[0],
        "Features": X.shape[1],
        "Target Classes": len(dataset.target_names),
        "Feature Names": list(dataset.feature_names),
        "Memory Usage (MB)": round(
            X.memory_usage(deep=True).sum() / 1024**2,
            2,
        ),
    }

    for key, value in overview.items():
        print(f"{key:<25}: {value}")

    print("\nTarget Names")

    for index, label in enumerate(dataset.target_names):
        print(f"  {index} -> {label}")

    print("\nFeature Data Types")

    print(X.dtypes)

    return overview


# =============================================================================
# Dataset Shape
# =============================================================================


def dataset_shape(dataset: Dataset) -> tuple[int, int]:
    """
    Return dataset shape.
    """

    rows, columns = dataset.features.shape

    logger.info(
        "Dataset Shape -> (%s, %s)",
        rows,
        columns,
    )

    return rows, columns


# =============================================================================
# Dataset Memory Usage
# =============================================================================


def memory_usage(dataset: Dataset) -> float:
    """
    Calculate dataset memory usage in MB.
    """

    memory = (
        dataset.features.memory_usage(
            deep=True
        ).sum()
        / 1024**2
    )

    logger.info(
        "Dataset Memory Usage %.2f MB",
        memory,
    )

    return memory


# =============================================================================
# Dataset Info
# =============================================================================


def dataset_info(dataset: Dataset) -> None:
    """
    Print pandas DataFrame information.
    """

    print_section("DataFrame Information")

    dataset.features.info()


# =============================================================================
# Column Names
# =============================================================================


def list_features(dataset: Dataset) -> list[str]:
    """
    Return feature names.
    """

    return list(dataset.feature_names)


# =============================================================================
# Preview Dataset
# =============================================================================


def preview_dataset(
    dataset: Dataset,
    rows: int = 5,
) -> pd.DataFrame:
    """
    Display first rows of dataset.
    """

    print_section("Dataset Preview")

    preview = dataset.features.head(rows)

    print(preview)

    return preview


# =============================================================================
# Main (temporary)
# =============================================================================

if __name__ == "__main__":

    from training.data_loader import load_dataset

    dataset = load_dataset()

    dataset_overview(dataset)

    dataset_shape(dataset)

    memory_usage(dataset)

    dataset_info(dataset)

    preview_dataset(dataset)
# =============================================================================
# Missing Value Analysis
# =============================================================================


def missing_value_analysis(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.

    Parameters
    ----------
    dataset:
        Loaded dataset.

    Returns
    -------
    pd.DataFrame
        Missing value report.
    """

    print_section("Missing Value Analysis")

    X = dataset.features

    missing_count = X.isnull().sum()

    missing_percentage = (
        missing_count / len(X) * 100
    )

    report = pd.DataFrame(
        {
            "feature": X.columns,
            "missing_count": missing_count.values,
            "missing_percentage": (
                missing_percentage.values
            ),
        }
    )

    report = report.sort_values(
        by="missing_count",
        ascending=False,
    )

    print(report)

    save_dataframe(
        report,
        "missing_values.csv",
    )

    total_missing = (
        report["missing_count"].sum()
    )

    if total_missing == 0:
        logger.info(
            "No missing values detected."
        )
    else:
        logger.warning(
            "Detected %s missing values.",
            total_missing,
        )

    return report



# =============================================================================
# Duplicate Analysis
# =============================================================================


def duplicate_analysis(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    Analyze duplicated samples.

    Parameters
    ----------
    dataset:
        Loaded dataset.

    Returns
    -------
    pd.DataFrame
        Duplicate report.
    """

    print_section("Duplicate Analysis")

    X = dataset.features

    duplicate_mask = X.duplicated()

    duplicate_count = (
        duplicate_mask.sum()
    )

    report = pd.DataFrame(
        {
            "metric": [
                "total_samples",
                "duplicate_samples",
                "duplicate_percentage",
            ],
            "value": [
                len(X),
                duplicate_count,
                round(
                    duplicate_count / len(X) * 100,
                    2,
                ),
            ],
        }
    )

    print(report)

    save_dataframe(
        report,
        "duplicate_analysis.csv",
    )

    if duplicate_count > 0:
        logger.warning(
            "Found %s duplicated samples.",
            duplicate_count,
        )
    else:
        logger.info(
            "No duplicated samples found."
        )

    return report



# =============================================================================
# Descriptive Statistics
# =============================================================================


def descriptive_statistics(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    Generate descriptive statistics.

    Includes:
    - count
    - mean
    - standard deviation
    - minimum
    - quartiles
    - maximum

    Parameters
    ----------
    dataset:
        Loaded dataset.

    Returns
    -------
    pd.DataFrame
        Statistical summary.
    """

    print_section(
        "Descriptive Statistics"
    )

    X = dataset.features


    statistics = X.describe().T


    # Add additional useful information

    statistics["median"] = (
        X.median()
    )

    statistics["range"] = (
        X.max()
        -
        X.min()
    )


    statistics = statistics[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "median",
            "75%",
            "max",
            "range",
        ]
    ]


    print(statistics)


    save_dataframe(
        statistics,
        "descriptive_statistics.csv",
    )


    return statistics



# =============================================================================
# Dataset Class Distribution Summary
# =============================================================================


def class_summary(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    Generate target class summary.

    Parameters
    ----------
    dataset:
        Loaded dataset.

    Returns
    -------
    pd.DataFrame
        Class distribution.
    """

    print_section(
        "Target Class Summary"
    )


    distribution = (
        dataset.target
        .value_counts()
        .sort_index()
    )


    percentage = (
        distribution
        /
        len(dataset.target)
        *
        100
    )


    report = pd.DataFrame(
        {
            "class_id": distribution.index,
            "samples": distribution.values,
            "percentage": percentage.values,
        }
    )


    report["class_name"] = (
        report["class_id"]
        .map(
            dict(
                enumerate(
                    dataset.target_names
                )
            )
        )
    )


    print(report)


    save_dataframe(
        report,
        "class_distribution.csv",
    )


    return report