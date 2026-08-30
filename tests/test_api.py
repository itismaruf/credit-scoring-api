# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "account_balance": 1,
    "duration_of_credit_monthly": 18,
    "payment_status_of_previous_credit": 4,
    "purpose": 2,
    "credit_amount": 1049,
    "value_savings_stocks": 1,
    "length_of_current_employment": 2,
    "instalment_per_cent": 4,
    "sex_marital_status": 2,
    "guarantors": 1,
    "duration_in_current_address": 4,
    "most_valuable_available_asset": 2,
    "age_years": 21,
    "concurrent_credits": 3,
    "type_of_apartment": 1,
    "no_of_credits_at_this_bank": 1,
    "occupation": 3,
    "no_of_dependents": 1,
    "telephone": 1,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert "creditability" in data
    assert "probability_good" in data
    assert "risk_level" in data

    assert data["creditability"] in (0, 1)
    assert 0.0 <= data["probability_good"] <= 1.0
    assert data["risk_level"] in ("low", "medium", "high")


def test_predict_missing_field():
    payload = VALID_PAYLOAD.copy()
    del payload["account_balance"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_range():
    payload = VALID_PAYLOAD.copy()
    payload["account_balance"] = 999  # за пределами ge=1, le=4

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_credit_amount():
    payload = VALID_PAYLOAD.copy()
    payload["credit_amount"] = -100

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_risk_level_consistency():
    """risk_level должен логически соответствовать probability_good"""
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    proba = data["probability_good"]
    risk = data["risk_level"]

    if proba >= 0.7:
        assert risk == "low"
    elif proba >= 0.4:
        assert risk == "medium"
    else:
        assert risk == "high"