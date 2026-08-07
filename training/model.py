"""
Machine Learning model module.

Responsibilities
----------------
- Create ML pipeline
- Train model


This module does NOT:
- Load datasets
- Split datasets
- Evaluate models
- Track MLflow
"""

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline


def build_model(
    preprocessing_pipeline,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Pipeline:
    """
    Build complete sklearn pipeline.

    Pipeline:

    Raw features
          |
          ↓
    StandardScaler
          |
          ↓
    RandomForest
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing_pipeline,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train,
    y_train,
) -> Pipeline:
    """
    Train sklearn pipeline.
    """

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline
