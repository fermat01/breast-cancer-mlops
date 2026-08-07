"""
Application configuration.
"""

import os

MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")


MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
