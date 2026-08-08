"""
Application logging configuration.
"""

import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    """
    Configure application-wide logging.
    """

    settings = get_settings()

    log_level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for a module.
    """

    return logging.getLogger(name)