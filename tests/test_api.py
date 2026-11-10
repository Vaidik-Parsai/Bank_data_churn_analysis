"""Unit and integration tests for the FastAPI churn prediction service."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with application lifespan triggered."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200 OK and reports model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_model_info_endpoint(client: TestClient):
    """Verify /model-info returns complete training metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "Random Forest" in data["model_name"]
    assert data["primary_metric"] == "roc_auc"
    assert data["cross_validation_roc_auc_mean"] > 0.80
    assert data["decision_threshold"] == 0.35
    assert len(data["feature_importances"]) > 0


def test_predict_single_valid_customer(client: TestClient):
    """Verify /predict returns valid probability and risk band for valid customer."""
    payload = {
        "CustomerId": 15634602,
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == 15634602
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert data["risk_band"] in ["Low", "Medium", "High"]
    assert data["threshold_used"] == 0.35


def test_predict_high_risk_customer(client: TestClient):
    """Customer matching high-risk demographics from EDA should be flagged with high risk."""
    payload = {
        "CustomerId": 99999999,
        "CreditScore": 500,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 55,
        "Tenure": 1,
        "Balance": 125000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 85000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] is True
    assert data["churn_probability"] >= 0.35
    assert data["risk_band"] in ["Medium", "High"]


def test_predict_validation_errors(client: TestClient):
    """Ensure invalid payloads trigger 422 Unprocessable Entity."""
    # 1. Invalid credit score (< 300)
    invalid_credit = {
        "CreditScore": 250,
        "Geography": "France",
        "Gender": "Male",
        "Age": 30,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }
    res = client.post("/predict", json=invalid_credit)
    assert res.status_code == 422

    # 2. Invalid age (< 18)
    invalid_age = invalid_credit.copy()
    invalid_age["CreditScore"] = 600
    invalid_age["Age"] = 16
    res = client.post("/predict", json=invalid_age)
    assert res.status_code == 422

    # 3. Invalid country (not in France, Germany, Spain)
    invalid_geo = invalid_credit.copy()
    invalid_geo["CreditScore"] = 600
    invalid_geo["Geography"] = "United States"
    res = client.post("/predict", json=invalid_geo)
    assert res.status_code == 422

    # 4. Missing required field
    missing_fields = {"CreditScore": 700}
    res = client.post("/predict", json=missing_fields)
    assert res.status_code == 422


def test_predict_batch_endpoint(client: TestClient):
    """Verify /predict/batch scores multiple customers accurately."""
    payload = {
        "customers": [
            {
                "CustomerId": 1,
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 30,
                "Tenure": 5,
                "Balance": 0.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 40000.0,
            },
            {
                "CustomerId": 2,
                "CreditScore": 550,
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 54,
                "Tenure": 2,
                "Balance": 110000.0,
                "NumOfProducts": 3,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 90000.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert 0.0 <= data["churn_rate_predicted"] <= 1.0


def test_predict_batch_empty_list_fails(client: TestClient):
    """Ensure empty batch fails validation with 422."""
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422

