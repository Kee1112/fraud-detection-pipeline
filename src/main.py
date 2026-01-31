from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.model import model
from src.features import build_feature_vector

app = FastAPI()


class Transaction(BaseModel):
    tx_count_user_1h: float = 0
    amount_vs_user_avg_24h: float = 0
    avg_amount_user_24h: float = 0
    amount_vs_merchant_avg: float = 0
    tx_count_merchant_1h: float = 0
    merchant_velocity_spike: float = 0
    merchant_fraud_rate: float = 0
    amount_log: float = 0
    is_high_amount: float = 0
    foreign_high_amount: float = 0
    high_amount_mobile: float = 0
    repeat_user_fast: float = 0
    merchant_risky_high_amount: float = 0



@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/predict")
def predict(tx: Transaction):
    X = build_feature_vector(tx.dict())

    prob = model.predict_proba(X)[0, 1]
    pred = int(prob > 0.5)

    return {
        "fraud_probability": float(prob),
        "is_fraud": pred
    }
