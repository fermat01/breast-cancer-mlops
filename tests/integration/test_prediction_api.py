"""
Integration tests for the prediction API.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app

# ============================================================
# Test client
# ============================================================

client = TestClient(app)


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
# Successful prediction
# ============================================================


@patch("app.api.routes.prediction.predict")
def test_create_prediction_success(
    mock_predict,
):
    """
    Test a successful prediction request.
    """

    mock_predict.return_value = {
        "prediction": 0,
        "prediction_label": "malignant",
        "probabilities": {
            "malignant": 0.97,
            "benign": 0.03,
        },
        "model_name": "breast-cancer-classifier",
        "model_alias": "champion",
        "model_version": "1",
        "model_run_id": "test-run-id",
    }

    response = client.post(
        "/api/v1/predictions",
        json={
            "features": VALID_FEATURES,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0

    assert data["prediction_label"] == "malignant"

    assert data["probabilities"] == {
        "malignant": 0.97,
        "benign": 0.03,
    }

    assert data["model_name"] == "breast-cancer-classifier"

    assert data["model_alias"] == "champion"

    assert data["model_version"] == "1"

    assert data["model_run_id"] == "test-run-id"

    mock_predict.assert_called_once_with(VALID_FEATURES)


# ============================================================
# Invalid feature count
# ============================================================


def test_create_prediction_invalid_feature_count():
    """
    Test that the API rejects a request containing
    an invalid number of features.
    """

    invalid_features = VALID_FEATURES[:29]

    response = client.post(
        "/api/v1/predictions",
        json={
            "features": invalid_features,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data


# ============================================================
# Missing features
# ============================================================


def test_create_prediction_missing_features():
    """
    Test that the API rejects a request without
    the features field.
    """

    response = client.post(
        "/api/v1/predictions",
        json={},
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data


# ============================================================
# Invalid feature type
# ============================================================


def test_create_prediction_invalid_feature_type():
    """
    Test that the API rejects non-numerical features.
    """

    invalid_features = VALID_FEATURES.copy()

    invalid_features[0] = "invalid"

    response = client.post(
        "/api/v1/predictions",
        json={
            "features": invalid_features,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data


# ============================================================
# Model unavailable
# ============================================================


@patch("app.api.routes.prediction.predict")
def test_create_prediction_model_unavailable(
    mock_predict,
):
    """
    Test that model/service failures return HTTP 503.
    """

    mock_predict.side_effect = RuntimeError("ML model has not been loaded.")

    response = client.post(
        "/api/v1/predictions",
        json={
            "features": VALID_FEATURES,
        },
    )

    assert response.status_code == 503

    data = response.json()

    assert data["detail"] == ("Prediction service is currently unavailable.")


# ============================================================
# Unexpected prediction error
# ============================================================


@patch("app.api.routes.prediction.predict")
def test_create_prediction_unexpected_error(
    mock_predict,
):
    """
    Test that unexpected prediction errors return HTTP 500.
    """

    mock_predict.side_effect = Exception("Unexpected model failure.")

    response = client.post(
        "/api/v1/predictions",
        json={
            "features": VALID_FEATURES,
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == (
        "An unexpected error occurred while generating " "the prediction."
    )
