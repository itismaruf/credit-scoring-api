from pydantic import BaseModel, ConfigDict, Field


class CreditApplication(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    account_balance: int = Field(..., ge=1, le=4)
    duration_of_credit_monthly: int = Field(..., gt=0)
    payment_status_of_previous_credit: int = Field(..., ge=0)  # 99 = unknown-код в данных
    purpose: int = Field(..., ge=0)
    credit_amount: float = Field(..., gt=0)
    value_savings_stocks: int = Field(..., ge=1)  # 99 = unknown-код в данных
    length_of_current_employment: int = Field(..., ge=1, le=5)
    instalment_per_cent: int = Field(..., ge=1, le=4)
    sex_marital_status: int = Field(..., ge=1, le=4)
    guarantors: int = Field(..., ge=1, le=3)
    duration_in_current_address: int = Field(..., ge=1, le=4)
    most_valuable_available_asset: int = Field(..., ge=1, le=4)
    age_years: int = Field(..., gt=0, lt=120)
    concurrent_credits: int = Field(..., ge=1)  # 99 = unknown-код в данных
    type_of_apartment: int = Field(..., ge=1, le=3)
    no_of_credits_at_this_bank: int = Field(..., ge=1)
    occupation: int = Field(..., ge=1)
    no_of_dependents: int = Field(..., ge=1)
    telephone: int = Field(..., ge=1, le=2)


class PredictionResponse(BaseModel):
    creditability: int
    probability_good: float
    risk_level: str