"""
Application configuration.

Configuration is loaded from environment variables and an optional
.env file for local development.

Production secrets should be provided through environment variables
or a secret-management system such as AWS Secrets Manager.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # ========================================================
    # Application
    # ========================================================

    app_name: str = "Breast Cancer Machine Learning API"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False

    # ========================================================
    # MLflow
    # ========================================================

    mlflow_tracking_uri: str = "http://mlflow:5000"

    model_name: str = "breast-cancer-classifier"
    model_alias: str = "champion"

    # ========================================================
    # MinIO / S3
    # ========================================================

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    mlflow_s3_endpoint_url: str | None = None

    # ========================================================
    # API
    # ========================================================

    api_prefix: str = "/api/v1"

    # ========================================================
    # Logging
    # ========================================================

    log_level: str = "INFO"

    # ========================================================
    # Pydantic configuration
    # ========================================================

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================
    # Computed values
    # ========================================================

    @property
    def model_uri(self) -> str:
        """
        Return the MLflow model URI using the configured alias.
        """

        return f"models:/{self.model_name}@{self.model_alias}"


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    The settings object is created once during the application
    lifecycle.
    """

    return Settings()
