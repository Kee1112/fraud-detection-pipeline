
# FastAPI inference app for Fraud Detection

from fastapi import FastAPI
from pydantic import BaseModel
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DoubleType
import os

# -------------------------
# Spark session
# -------------------------
spark = SparkSession.builder \
    .appName("FraudInferenceAPI") \ #additional change master wasnt there
    .master("local[*]")\
    .getOrCreate()

# -------------------------
# Load saved models
# -------------------------
#additional change was model_dir = "fraud-detection-pipleine/models"
MODEL_DIR = "/app/models"

assembler = VectorAssembler.load(
    f"{MODEL_DIR}/assembler"
)

scaler = StandardScalerModel.load(
    f"{MODEL_DIR}/scaler"
)

gbt_model = GBTClassificationModel.load(
    f"{MODEL_DIR}/gbt_fraud_model"
)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Fraud Detection API")

# -------------------------
# Request schema
# -------------------------
class Transaction(BaseModel):
    tx_count_user_1h: float
    amount_vs_user_avg_24h: float
    avg_amount_user_24h: float
    amount_vs_merchant_avg: float
    tx_count_merchant_1h: float
    merchant_velocity_spike: float
    merchant_fraud_rate: float
    amount_log: float
    is_high_amount: float
    foreign_high_amount: float
    high_amount_mobile: float
    repeat_user_fast: float
    merchant_risky_high_amount: float

# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Model metadata
# -------------------------
@app.get("/model-info")
def model_info():
    return {
        "model": "Gradient Boosted Trees",
        "framework": "PySpark ML",
        "features": 13,
        "threshold": 0.5
    }

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(transaction: Transaction):
  feature_cols = [
    "tx_count_user_1h",
    "amount_vs_user_avg_24h",
    "avg_amount_user_24h",
    "amount_vs_merchant_avg",
    "tx_count_merchant_1h",
    "merchant_velocity_spike",
    "merchant_fraud_rate",
    "amount_log",
    "is_high_amount",
    "foreign_high_amount",
    "high_amount_mobile",
    "repeat_user_fast",
    "merchant_risky_high_amount"
]

  schema = StructType([
      StructField(c, DoubleType(), True) for c in feature_cols
  ])
  row = transaction.dict()
  values = [[float(row[c]) for c in feature_cols]]


  df = spark.createDataFrame(values, schema=schema)


  df = assembler.transform(df)
  df = scaler.transform(df)

  preds = gbt_model.transform(df)

  prob = preds.select("probability").first()[0][1]

  return {
        "fraud_probability": float(prob),
        "is_fraud": int(prob > 0.5)
    }
