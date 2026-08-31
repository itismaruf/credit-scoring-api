# Credit Scoring API

A machine learning service for predicting credit risk (good / bad) for loan applicants, built end-to-end from exploratory data analysis and model experimentation to a containerized and deployed REST API.

**Live API:** https://credit-scoring-api-production-17f6.up.railway.app
**Swagger UI:** https://credit-scoring-api-production-17f6.up.railway.app/docs

## Overview

This project implements a binary credit scoring model using the German Credit Data (Statlog) dataset, with a focus on recall for the bad credit class. In lending, failing to identify a high-risk applicant can be more costly than rejecting a low-risk applicant.

Several models and approaches were evaluated, with tuned CatBoost selected as the final model. The trained model is served through a FastAPI REST API, containerized with Docker, and deployed on Railway.

## Pipeline

```
EDA → Baseline Modeling → CatBoost → Class Weight Experiments → Hyperparameter Tuning
   → SHAP Interpretation → Feature Engineering → Final CatBoost Model
   → FastAPI → Docker → Railway Deployment
```

## Dataset

The project uses the German Credit Data (Statlog) dataset:

- 1,000 applicants
- 19 features (categorical and numerical)
- binary target `creditability`: 1 = good, 0 = bad
- approximately 700 good and 300 bad applicants

The `foreign_worker` feature was removed during EDA as a protected attribute for fairness considerations, rather than because of data leakage.

## Model Performance

The primary metric for model comparison was recall for the bad class, with precision and ROC-AUC used as additional evaluation metrics.

| Model | Recall (bad) | Precision (bad) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline, balanced) | 0.77 | 0.54 | 0.8294 |
| CatBoost (default) | 0.43 | 0.67 | 0.8324 |
| CatBoost (balanced class weights) | 0.67 | 0.62 | 0.8285 |
| **CatBoost (tuned, final model)** | **0.67** | **0.62** | **0.8335** |

Hyperparameters were tuned using `RandomizedSearchCV` with:
- 5-fold cross-validation
- 30 iterations

SHAP analysis identified the following as the strongest predictors:
- `account_balance`
- `value_savings_stocks`
- `duration_of_credit_monthly`

These features are also consistent with domain intuition for credit scoring, as account balance, savings, and loan duration can provide important information about an applicant's financial situation and credit risk.

## Project Structure

```text
credit-scoring-api/
├── app/
│   ├── main.py             # FastAPI application and API endpoints
│   ├── schemas.py          # Pydantic request/response models
│   ├── model_loader.py     # CatBoost model loading and caching
│   └── config.py           # Paths, thresholds, and feature configuration
│
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed dataset
│
├── models/                 # Trained CatBoost model (.cbm)
│
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory data analysis
│   └── 02_modeling.ipynb   # Baselines, CatBoost, tuning, SHAP
│
├── tests/
│   └── test_api.py         # API tests
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Running Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the application:
```bash
python -m app.main
```

The API will be available at `http://127.0.0.1:8000`, with interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

## Running with Docker

Build and start the application using Docker Compose:
```bash
docker compose up --build
```

The API will then be available at `http://127.0.0.1:8000`.

## API

### POST /predict

Accepts information about a loan applicant and returns a creditability prediction, probability, and risk level.

**Example request:**
```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
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
    "telephone": 1
  }'
```

**Example response:**
```json
{
  "creditability": 1,
  "probability_good": 0.82,
  "risk_level": "low"
}
```

### GET /health

Health check endpoint used to verify that the service is running.

## Live Demo

The API is deployed as a Docker container on Railway and is publicly accessible.

**Live API:** https://credit-scoring-api-production-17f6.up.railway.app
**Interactive Swagger documentation:** https://credit-scoring-api-production-17f6.up.railway.app/docs

The deployed Swagger interface can be used to send real requests to the model and receive predictions.

## Tests

Run the test suite with:
```bash
pytest tests/ -v
```

Tests cover:
- health check
- valid payloads
- invalid payloads
- field validation
- required fields
- consistency between `risk_level` and prediction thresholds

## Tech Stack

Python, pandas, scikit-learn, CatBoost, SHAP, FastAPI, Pydantic, pytest, Docker, Railway