from fastapi.testclient import TestClient
from main import app, BreastCancerFeatures

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_prediction():
    test_data = {
        "mean_radius": 14.5,
        "mean_texture": 20.0,
        "mean_perimeter": 90.0,
        "mean_area": 600.0,
        "mean_smoothness": 0.1,
        "mean_compactness": 0.15,
        "mean_concavity": 0.15,
        "mean_concave_points": 0.05,
        "mean_symmetry": 0.2,
        "mean_fractal_dimension": 0.06,
        "radius_error": 0.3,
        "texture_error": 1.0,
        "perimeter_error": 2.0,
        "area_error": 30.0,
        "smoothness_error": 0.005,
        "compactness_error": 0.02,
        "concavity_error": 0.02,
        "concave_points_error": 0.01,
        "symmetry_error": 0.02,
        "fractal_dimension_error": 0.003,
        "worst_radius": 18.0,
        "worst_texture": 27.0,
        "worst_perimeter": 115.0,
        "worst_area": 850.0,
        "worst_smoothness": 0.15,
        "worst_compactness": 0.3,
        "worst_concavity": 0.3,
        "worst_concave_points": 0.1,
        "worst_symmetry": 0.3,
        "worst_fractal_dimension": 0.08
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["malignant", "benign"]
