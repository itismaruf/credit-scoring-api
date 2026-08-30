from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "credit_scoring_catboost.cbm"

API_TITLE = "Credit Scoring API"
API_VERSION = "1.0.0"

# пороги для risk_level, можно потюнить позже
RISK_THRESHOLDS = {
    "low": 0.7,     # probability_good >= 0.7
    "medium": 0.4,  # 0.4 <= probability_good < 0.7
    # ниже 0.4 — high
}

# порядок колонок должен точно совпадать с X_train при обучении в 02_modeling.ipynb
TRAIN_COLUMN_ORDER = [
    "account_balance",
    "duration_of_credit_monthly",
    "payment_status_of_previous_credit",
    "purpose",
    "credit_amount",
    "value_savings_stocks",
    "length_of_current_employment",
    "instalment_per_cent",
    "sex_marital_status",
    "guarantors",
    "duration_in_current_address",
    "most_valuable_available_asset",
    "age_years",
    "concurrent_credits",
    "type_of_apartment",
    "no_of_credits_at_this_bank",
    "occupation",
    "no_of_dependents",
    "telephone",
    "credit_amount_per_month",
]