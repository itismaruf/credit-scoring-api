# Credit Scoring API

ML-сервис для оценки кредитоспособности заёмщика, построенный на German Credit Data — от разведочного анализа данных и экспериментов с моделями до контейнеризированного REST API.

**Live API:** https://credit-scoring-api-production-17f6.up.railway.app
**Swagger UI:** https://credit-scoring-api-production-17f6.up.railway.app/docs

## Описание

Проект реализует бинарную модель кредитного скоринга с фокусом на **Recall для класса bad**. В кредитном скоринге пропуск рискованного заёмщика может быть более дорогим, чем отказ надёжному клиенту, поэтому качество обнаружения класса bad является одним из ключевых критериев оценки.

Финальная модель — CatBoost, обученная после сравнения с baseline-моделью и последующего гиперпараметрического тюнинга. Модель сохранена и интегрирована в FastAPI-сервис, который предоставляет prediction через REST API.

## ML-пайплайн

```
EDA → Baseline models → CatBoost → Class weights → Hyperparameter tuning
   → SHAP interpretation → Feature engineering experiments → Final CatBoost model
   → FastAPI → Docker → Railway deployment
```

## Датасет

Используется German Credit Data (Statlog):

- 1000 заявителей
- 19 признаков (категориальные и числовые)
- бинарный target `creditability`: 1 — good, 0 — bad
- распределение классов: примерно 700 good / 300 bad

Колонка `foreign_worker` была удалена на этапе EDA как защищённый атрибут — по соображениям справедливости, а не из-за утечки данных.

## Качество моделей

Основным критерием при сравнении моделей был Recall для класса bad, дополнительно оценивались Precision и ROC-AUC.

| Модель | Recall (bad) | Precision (bad) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline, balanced) | 0.77 | 0.54 | 0.8294 |
| CatBoost (default) | 0.43 | 0.67 | 0.8324 |
| CatBoost (balanced class weights) | 0.67 | 0.62 | 0.8285 |
| **CatBoost (tuned, final)** | **0.67** | **0.62** | **0.8335** |

Гиперпараметры CatBoost подбирались с помощью `RandomizedSearchCV`:
- 5-fold cross-validation
- 30 итераций
- оптимизация с учётом выбранной метрики

## Интерпретация модели

Для интерпретации предсказаний использовался SHAP.

Наиболее значимые признаки:
- `account_balance`
- `value_savings_stocks`
- `duration_of_credit_monthly`

Их влияние соответствует бизнес-интуиции кредитного скоринга: финансовое положение клиента и параметры кредита существенно связаны с вероятностью дефолта.

Эксперименты с feature engineering, включая удаление слабых по SHAP признаков и добавление ручных взаимодействий, не дали значимого улучшения относительно tuned CatBoost. Это также согласуется со способностью деревьев CatBoost самостоятельно находить нелинейные взаимодействия между признаками.

## REST API

### POST /predict

Принимает данные кредитной заявки и возвращает:
- `creditability` — предсказанный класс
- `probability_good` — вероятность хорошей кредитоспособности
- `risk_level` — интерпретированный уровень риска

**Пример запроса:**
```json
{
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
}
```

**Пример ответа:**
```json
{
  "creditability": 1,
  "probability_good": 0.82,
  "risk_level": "low"
}
```

### GET /health

Health check для проверки доступности сервиса.

## Swagger

API имеет автоматически сгенерированную интерактивную документацию FastAPI.

**Live Swagger:** https://credit-scoring-api-production-17f6.up.railway.app/docs

Через Swagger можно отправлять реальные запросы к задеплоенной модели и получать predictions.

## Структура проекта

```text
credit-scoring-api/
├── app/
│   ├── main.py             # FastAPI-приложение и API endpoints
│   ├── schemas.py          # Pydantic-модели запросов и ответов
│   ├── model_loader.py     # Загрузка и кэширование CatBoost-модели
│   └── config.py           # Конфигурация, пути и параметры модели
│
├── data/
│   ├── raw/                # Исходный датасет
│   └── processed/          # Обработанные данные
│
├── models/
│   └── *.cbm                # Сохранённая CatBoost-модель
│
├── notebooks/
│   ├── 01_EDA.ipynb        # EDA
│   └── 02_modeling.ipynb   # Modeling, tuning и SHAP
│
├── tests/
│   └── test_api.py         # API-тесты
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Запуск локально

```bash
pip install -r requirements.txt
python -m app.main
```

После запуска:
- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

## Запуск через Docker

```bash
docker compose up --build
```

После запуска API будет доступен на `http://127.0.0.1:8000`, Swagger — на `http://127.0.0.1:8000/docs`.

## Тесты

```bash
pytest tests/ -v
```

Тесты покрывают:
- health check
- валидные запросы
- невалидные запросы
- Pydantic validation
- проверку обязательных полей
- согласованность `risk_level` с установленными порогами

## Deployment

Приложение контейнеризировано с помощью Docker и развёрнуто на Railway.

```
GitHub → Docker → Railway → FastAPI → CatBoost → /predict
```

**Live API:** https://credit-scoring-api-production-17f6.up.railway.app
**Swagger:** https://credit-scoring-api-production-17f6.up.railway.app/docs

## Технологии

Python, pandas, scikit-learn, CatBoost, SHAP, FastAPI, Pydantic, pytest, Docker, Railway