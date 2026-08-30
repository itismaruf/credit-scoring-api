# Credit Scoring API

A machine learning service that predicts credit risk (good/bad) for loan applicants, built end-to-end from EDA to a containerized REST API.

## Overview

This project implements a binary credit scoring model on the German Credit Data dataset, with a focus on **recall for the "bad" credit class** — in lending, approving a bad loan is more costly than rejecting a good applicant. The final model is served through a FastAPI application, containerized with Docker.

**Pipeline:** EDA → baseline modeling → CatBoost tuning → SHAP interpretation → feature engineering → FastAPI service → Docker

## Dataset

German Credit Data (Statlog), 1000 applicants, 19 features (categorical + numeric) plus target `creditability` (1 = good, 0 = bad). Class distribution is imbalanced (~700 good / ~300 bad).

The `foreign_worker` column was dropped during EDA as a protected attribute, for fairness reasons rather than data leakage.

## Model Performance

| Model | Recall (bad) | Precision (bad) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline, balanced) | 0.77 | 0.54 | 0.8294 |
| CatBoost (default) | 0.43 | 0.67 | 0.8324 |
| CatBoost (balanced class weights) | 0.67 | 0.62 | 0.8285 |
| **CatBoost (tuned, final model)** | **0.67** | **0.62** | **0.8335** |

Hyperparameters were tuned via `RandomizedSearchCV` (5-fold CV, 30 iterations). SHAP analysis identified `account_balance`, `value_savings_stocks`, and `duration_of_credit_monthly` as the strongest predictors — consistent with domain intuition for credit scoring.

## Project Structure
```text
credit-scoring-api/
├── app/
│   ├── main.py           # FastAPI-приложение, эндпоинты /predict и /health
│   ├── schemas.py        # Pydantic-модели запроса/ответа
│   ├── model_loader.py   # Загрузка и кэширование модели CatBoost
│   └── config.py         # Пути, пороги, порядок колонок
├── data/
│   ├── raw/               # Исходный датасет
│   └── processed/         # Обработанный датасет для обучения
├── models/                 # Обученная модель CatBoost (.cbm)
├── notebooks/
│   ├── 01_EDA.ipynb        # Разведочный анализ данных
│   └── 02_modeling.ipynb   # Baseline, CatBoost, тюнинг, SHAP
├── tests/
│   └── test_api.py         # Тесты API (pytest)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```


## Running Locally

```bash
pip install -r requirements.txt
python -m app.main
```

API available at `http://127.0.0.1:8000`, interactive docs at `http://127.0.0.1:8000/docs`.

## Running with Docker

```bash
docker compose up --build
```

## API Usage

**POST /predict**

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

Response:
```json
{
  "creditability": 1,
  "probability_good": 0.82,
  "risk_level": "low"
}
```

**GET /health** — service liveness check.

## Tests

```bash
pytest tests/ -v
```

Covers health check, valid/invalid payloads, field validation, and risk-level threshold consistency.

## Tech Stack

Python, pandas, scikit-learn, CatBoost, SHAP, FastAPI, Pydantic, pytest, Docker

## Notes

- `occupation`, `payment_status_of_previous_credit`, `value_savings_stocks`, and `concurrent_credits` contain a code value of `99` in the source data (likely "unknown"); this is preserved as a valid category rather than imputed, since CatBoost handles it natively as a category and it affects a small fraction of records.
- Feature engineering experiments (removing low-SHAP-importance features, adding a manual interaction feature) did not meaningfully improve metrics beyond the tuned baseline — consistent with CatBoost's ability to learn interactions natively on tree splits.