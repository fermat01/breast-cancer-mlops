"""
Training package for the Breast Cancer MLOps project.

This package contains everything related to the machine learning
training pipeline, including:

- Data loading
- Data validation
- Data preprocessing
- Dataset splitting
- Model training
- Model evaluation
- MLflow experiment tracking

The API application imports only the trained model artifacts.
Training code should never depend on FastAPI.
"""