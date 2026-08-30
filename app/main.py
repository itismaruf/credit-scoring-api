from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Depends

from app.config import API_TITLE, API_VERSION, RISK_THRESHOLDS, TRAIN_COLUMN_ORDER
from app.model_loader import load_model, get_model
from app.schemas import CreditApplication, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)


def get_risk_level(probability_good: float) -> str:
    if probability_good >= RISK_THRESHOLDS["low"]:
        return "low"
    elif probability_good >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "high"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication, model=Depends(get_model)):
    input_df = pd.DataFrame([application.model_dump()])

    # восстанавливаем производную фичу, как в тренировке
    input_df["credit_amount_per_month"] = (
        input_df["credit_amount"] / input_df["duration_of_credit_monthly"]
    )

    input_df = input_df[TRAIN_COLUMN_ORDER]

    proba_good = model.predict_proba(input_df)[0][1]
    prediction = int(proba_good >= 0.5)

    return PredictionResponse(
        creditability=prediction,
        probability_good=round(float(proba_good), 4),
        risk_level=get_risk_level(proba_good)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)