from fastapi.testclient import TestClient
import pytest
from src.mlops_churn_api import app


@pytest.fixture
def client():
    """Create a test client with the application lifespan enabled."""
    with TestClient(app) as client:
        yield client


def test_root(client):
    """Test that the root endpoint returns a successful response."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Churn API is running"}


def test_health(client):
    """Test that the health endpoint reports the API and model as healthy."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict(client):
    """Test that the prediction endpoint returns a valid churn prediction."""
    data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "probability" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1
