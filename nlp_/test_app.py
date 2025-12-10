# Simple unit tests
"""
Unit and Integration Tests for Model API

"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    """Test client fixture"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


#  UNIT TESTS


def test_home_endpoint(client):
    """Test home endpoint returns success"""
    response = client.get("/")
    assert response.status_code == 200


def test_predict_endpoint_exists(client):
    """Test predict endpoint is accessible"""
    response = client.post(
        "/predict", json={"text": "test"}, content_type="application/json"
    )
    assert response.status_code == 200


def test_predict_returns_json(client):
    """Test predict returns valid JSON"""
    response = client.post(
        "/predict", json={"text": "Sample legal text"}, content_type="application/json"
    )
    data = json.loads(response.data)
    assert "prediction" in data
    assert "confidence" in data


def test_confidence_is_valid(client):
    """Test confidence score is between 0 and 1"""
    response = client.post(
        "/predict",
        json={"text": "Legal document text"},
        content_type="application/json",
    )
    data = json.loads(response.data)
    assert 0 <= data["confidence"] <= 1


# INTEGRATION TESTS
def test_metrics_endpoint(client):
    """Integration test: metrics endpoint works after predictions"""
    # Make a prediction first
    client.post("/predict", json={"text": "test"})
    # Check metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "total_predictions" in data


def test_retrain_trigger(client):
    """Integration test: manual retrain endpoint"""
    response = client.post("/trigger_retrain")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "status" in data
