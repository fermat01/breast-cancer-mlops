"""
Application configuration.
"""
"""
import os
MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
"""


from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    app_name: str = "Breast Cancer ML API"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow_tracking_uri: str = "http://mlflow:5000"

    model_name: str = "breast-cancer-classifier"
    model_alias: str = "champion"

    # --------------------------------------------------------
    # MinIO / S3
    # --------------------------------------------------------

    aws_access_key_id: str = "minio"
    aws_secret_access_key: str = "mlflow77"

    mlflow_s3_endpoint_url: str = "http://minio:9000"

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    api_prefix: str = "/api/v1"

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_level: str = "INFO"

    # --------------------------------------------------------
    # Pydantic configuration
    # --------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --------------------------------------------------------
    # Computed values
    # --------------------------------------------------------

    @property
    def model_uri(self) -> str:
        """
        MLflow model URI using the configured alias.
        """

        return f"models:/{self.model_name}@{self.model_alias}"


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    The settings object is created only once during
    the application lifecycle.
    """

    return Settings()