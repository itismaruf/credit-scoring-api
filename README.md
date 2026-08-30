# Credit Scoring API

ML-сервис, предсказывающий кредитоспособность заёмщика (хороший/плохой риск), построенный от EDA до контейнеризированного REST API.

## Описание

Проект реализует бинарную модель кредитного скоринга на датасете German Credit Data, с упором на **recall для класса "bad"** — в кредитовании выдать плохой кредит дороже, чем отказать хорошему заёмщику. Финальная модель обёрнута в FastAPI-сервис, контейнеризирована через Docker.

**Пайплайн:** EDA → baseline-модели → тюнинг CatBoost → интерпретация через SHAP → feature engineering → FastAPI-сервис → Docker

## Датасет

German Credit Data (Statlog), 1000 заявителей, 19 фичей (категориальные + числовые) плюс таргет `creditability` (1 = good, 0 = bad). Классы несбалансированы (~700 good / ~300 bad).

Колонка `foreign_worker` была удалена на этапе EDA как защищённый атрибут — по соображениям справедливости, а не как утечка данных.

## Качество моделей

| Модель | Recall (bad) | Precision (bad) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline, balanced) | 0.77 | 0.54 | 0.8294 |
| CatBoost (default) | 0.43 | 0.67 | 0.8324 |
| CatBoost (balanced class weights) | 0.67 | 0.62 | 0.8285 |
| **CatBoost (tuned, финальная модель)** | **0.67** | **0.62** | **0.8335** |

Гиперпараметры подобраны через `RandomizedSearchCV` (5-fold CV, 30 итераций). SHAP-анализ показал самых сильных предикторов: `account_balance`, `value_savings_stocks`, `duration_of_credit_monthly` — совпадает с бизнес-интуицией скоринга.

## Структура проекта
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

## Запуск локально

```bash
pip install -r requirements.txt
python -m app.main
```

API доступен на `http://127.0.0.1:8000`, интерактивная документация — `http://127.0.0.1:8000/docs`.

## Запуск через Docker

```bash
docker compose up --build
```

## Использование API

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

Ответ:
```json
{
  "creditability": 1,
  "probability_good": 0.82,
  "risk_level": "low"
}
```

**GET /health** — проверка живости сервиса.

## Тесты

```bash
pytest tests/ -v
```

Покрывают: health check, валидные/невалидные запросы, валидацию полей, согласованность risk_level с порогами.

## Стек технологий

Python, pandas, scikit-learn, CatBoost, SHAP, FastAPI, Pydantic, pytest, Docker

## Примечания

- `occupation`, `payment_status_of_previous_credit`, `value_savings_stocks` и `concurrent_credits` содержат код `99` в исходных данных (вероятно, "неизвестно"); это значение сохранено как валидная категория, а не импутировано — CatBoost сам обрабатывает его как отдельную категорию, и оно затрагивает малую долю записей.
- Эксперименты с feature engineering (удаление слабых по SHAP фич, добавление ручной интеракции) не дали значимого улучшения метрик сверх tuned-baseline — что согласуется со способностью CatBoost находить взаимодействия фич самостоятельно через сплиты деревьев.
