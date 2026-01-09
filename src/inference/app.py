from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import col
import uvicorn

# ------------------------
# Spark session
# ------------------------
spark = SparkSession.builder \
    .appName("FraudInferenceAPI") \
    .getOrCreate()

# ------------------------
# Load artifacts
# ------------------------
ASSEMBLER_PATH = "fraud-detection-pipeline/models/assembler"
SCALER_PATH = "fraud-detection-pipeline/models/scaler"
MODEL_PATH = "fraud-detection-pipeline/models/gbt_fraud_model"

assembler = VectorAssembler.load(ASSEMBLER_PATH)
scaler = StandardScalerModel.load(SCALER_PATH)
model = GBTClassificationModel.load(MODEL_PATH)

# Feature order MUST match training
FEATURE_COLS = assembler.getInputCols()

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Fraud Detection API")

# ------------------------
# Input schema
# ------------------------
class Transaction(BaseModel):
    tx_count_user_1h: float
    amount_vs_user_avg_24h: float
    avg_amount_user_24h: float
    amount_vs_merchant_avg: float
    tx_count_merchant_1h: float
    merchant_velocity_spike: int
    merchant_fraud_rate: float
    amount_log: float
    is_high_amount: int
    foreign_high_amount: int
    high_amount_mobile: int
    repeat_user_fast: int
    merchant_risky_high_amount: int

# ------------------------
# Prediction endpoint
# ------------------------
@app.post("/predict")
def predict(tx: Transaction):
    data = [{col: getattr(tx, col) for col in FEATURE_COLS}]
    df = spark.createDataFrame(data)

    df = assembler.transform(df)
    df = scaler.transform(df)
    preds = model.transform(df)

    result = preds.select(
        col("probability")[1].alias("fraud_probability"),
        col("prediction")
    ).collect()[0]

    return {
        "fraud_probability": float(result["fraud_probability"]),
        "prediction": int(result["prediction"])
    }


# ------------------------
# Run locally
# ------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
