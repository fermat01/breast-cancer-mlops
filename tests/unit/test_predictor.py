"""
Unit tests for the prediction service.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.predictor import predict

# ============================================================
# Test data
# ============================================================

VALID_FEATURES = [
    17.99,
    10.38,
    122.8,
    1001.0,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189,
]


# ============================================================
# Mock model
# ============================================================


def create_mock_model():
    """
    Create a mock MLflow/sklearn model.
    """

    model = MagicMock()

    # --------------------------------------------------------
    # Mock prediction
    # --------------------------------------------------------

    model.predict.return_value = np.array([0])

    # --------------------------------------------------------
    # Mock underlying sklearn model
    # --------------------------------------------------------

    sklearn_model = model._model_impl.sklearn_model

    sklearn_model.classes_ = np.array([0, 1])

    sklearn_model.predict_proba.return_value = np.array([[0.97, 0.03]])

    return model


# ============================================================
# Mock model metadata
# ============================================================


def create_mock_metadata():
    """
    Create mock MLflow model metadata.
    """

    metadata = MagicMock()

    metadata.name = "breast-cancer-classifier"
    metadata.alias = "champion"
    metadata.version = "1"
    metadata.run_id = "test-run-id"
    metadata.source = "test-source"

    return metadata


# ============================================================
# Prediction tests
# ============================================================


@patch("app.services.predictor.get_model_metadata")
@patch("app.services.predictor.get_model")
def test_predict_returns_prediction(
    mock_get_model,
    mock_get_model_metadata,
):
    """
    Test that predict() returns the predicted class.
    """

    mock_get_model.return_value = create_mock_model()
    mock_get_model_metadata.return_value = create_mock_metadata()

    result = predict(VALID_FEATURES)

    assert result["prediction"] == 0


@patch("app.services.predictor.get_model_metadata")
@patch("app.services.predictor.get_model")
def test_predict_returns_prediction_label(
    mock_get_model,
    mock_get_model_metadata,
):
    """
    Test that the predicted class is mapped to
    the correct human-readable label.
    """

    mock_get_model.return_value = create_mock_model()
    mock_get_model_metadata.return_value = create_mock_metadata()

    result = predict(VALID_FEATURES)

    assert result["prediction_label"] == "malignant"


@patch("app.services.predictor.get_model_metadata")
@patch("app.services.predictor.get_model")
def test_predict_returns_probabilities(
    mock_get_model,
    mock_get_model_metadata,
):
    """
    Test that prediction probabilities are correctly
    extracted and mapped to class labels.
    """

    mock_get_model.return_value = create_mock_model()
    mock_get_model_metadata.return_value = create_mock_metadata()

    result = predict(VALID_FEATURES)

    assert result["probabilities"] == {
        "malignant": 0.97,
        "benign": 0.03,
    }


# ============================================================
# DataFrame tests
# ============================================================


@patch("app.services.predictor.get_model_metadata")
@patch("app.services.predictor.get_model")
def test_predict_calls_model_with_dataframe(
    mock_get_model,
    mock_get_model_metadata,
):
    """
    Test that the model receives a pandas DataFrame
    containing exactly 30 features.
    """

    mock_model = create_mock_model()

    mock_get_model.return_value = mock_model
    mock_get_model_metadata.return_value = create_mock_metadata()

    predict(VALID_FEATURES)

    mock_model.predict.assert_called_once()

    dataframe = mock_model.predict.call_args.args[0]

    assert isinstance(dataframe, pd.DataFrame)

    assert dataframe.shape == (1, 30)


# ============================================================
# Feature validation
# ============================================================


def test_predict_rejects_invalid_feature_count():
    """
    Test that predict() rejects an invalid number
    of features.
    """

    invalid_features = VALID_FEATURES[:29]

    with pytest.raises(
        ValueError,
        match="Expected 30",
    ):
        predict(invalid_features)


# ============================================================
# Model availability
# ============================================================


@patch("app.services.predictor.get_model")
def test_predict_raises_when_model_unavailable(
    mock_get_model,
):
    """
    Test that model loading failures are propagated.
    """

    mock_get_model.side_effect = RuntimeError("ML model has not been loaded.")

    with pytest.raises(
        RuntimeError,
        match="ML model has not been loaded",
    ):
        predict(VALID_FEATURES)


# ============================================================
# Metadata
# ============================================================


@patch("app.services.predictor.get_model_metadata")
@patch("app.services.predictor.get_model")
def test_predict_returns_model_metadata(
    mock_get_model,
    mock_get_model_metadata,
):
    """
    Test that MLflow model metadata is included
    in the prediction result.
    """

    mock_get_model.return_value = create_mock_model()
    mock_get_model_metadata.return_value = create_mock_metadata()

    result = predict(VALID_FEATURES)

    assert result["model_name"] == "breast-cancer-classifier"
    assert result["model_alias"] == "champion"
    assert result["model_version"] == "1"
    assert result["model_run_id"] == "test-run-id"
