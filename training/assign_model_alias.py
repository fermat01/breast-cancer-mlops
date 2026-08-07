"""
Assign MLflow model alias.

Modern replacement for MLflow stages.

Old:
    Production stage

New:
    champion alias
"""

from training.mlflow_tracker import (
    set_model_alias,
)

MODEL_VERSION = 1


if __name__ == "__main__":

    result = set_model_alias(
        version=MODEL_VERSION,
        alias="champion",
    )

    print("Model alias assigned successfully:")

    print(result)
