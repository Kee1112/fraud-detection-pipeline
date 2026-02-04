# Fraud Detection API (FastAPI + Spark Training + sklearn Inference)

Production‑ready fraud detection service built with **PySpark for feature engineering & training** and **FastAPI + sklearn for lightweight cloud inference**.

This project demonstrates how to move from a distributed ML training pipeline to a deployable, low‑latency prediction API suitable for cloud platforms like Render.

---

#  Overview

This system detects potentially fraudulent transactions using engineered behavioral and merchant‑risk features.

Training is performed with **Spark ML** on parquet feature data, while inference is served using a **sklearn Gradient Boosting model** and a manually exported scaler — avoiding Spark at runtime for reliability and speed.

---

#  ML Pipeline Design

## Training (Spark)

* Load parquet transaction feature dataset
* Feature assembly with Spark VectorAssembler
* Standard scaling with Spark StandardScaler
* Train models:

  * Logistic Regression
  * Gradient Boosted Trees (Spark)
* Export scaler statistics (mean/std) → JSON
* Sample data → convert to Pandas → train sklearn GBT
* Save sklearn model for inference

## Inference (API)

* FastAPI REST service
* Manual feature vector builder
* Manual standard scaling using exported scaler stats
* sklearn GradientBoostingClassifier for prediction
* Probability + binary fraud flag returned

This avoids JVM/Spark dependencies in production.

---

# 🗂 Project Structure



---

#This app is live at https://fraud-detection-pipeline-1-uag0.onrender.com 
```
feel free to check it out :)
to test out the predict end point 
https://fraud-detection-pipeline-1-uag0.onrender.com/docs 
```

# ⚙️ Installation (Local)

## 1️⃣ Clone repo

```bash
git clone <this-repo-url>
cd fraud-detection-pipeline
```

## 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run API

```bash
uvicorn src.main:app --reload
```

---

# 🧪 API Usage

## Health Check

```
GET /health
```

Response:

```json
{"status": "ok"}
```

---

## Prediction

```
POST /predict
```

### Example Request

```json
{
  "tx_count_user_1h": 2,
  "amount_vs_user_avg_24h": 1.3,
  "avg_amount_user_24h": 120,
  "amount_vs_merchant_avg": 0.9,
  "tx_count_merchant_1h": 4,
  "merchant_velocity_spike": 0,
  "merchant_fraud_rate": 0.01,
  "amount_log": 4.2,
  "is_high_amount": 1,
  "foreign_high_amount": 0,
  "high_amount_mobile": 1,
  "repeat_user_fast": 0,
  "merchant_risky_high_amount": 0
}
```

### Example Response

```json
{
  "fraud_probability": 0.73,
  "is_fraud": 1
}
```

---

# 🚀 Deployment (Render)

Create a **Render Web Service** connected to this repo.

### Build Command

```
pip install -r requirements.txt
```

### Start Command

```
uvicorn src.main:app --host 0.0.0.0 --port 10000
```

After deployment:

```
https://<service>.onrender.com/docs
```

---

# ⚠️ Why sklearn for Inference Instead of Spark

Running Spark inside web APIs causes:

* JVM memory pressure
* container crashes
* token/accumulator errors
* cold start delays

Using sklearn for inference gives:

✅ fast startup
✅ low memory usage
✅ cloud compatibility
✅ simpler deployment

Industry standard pattern: **Distributed training → lightweight inference service**.

---

# 📊 Features Used

* User transaction velocity
* Merchant velocity spikes
* Relative transaction amount
* Merchant fraud rate
* High‑amount flags
* Behavioral repeat patterns

All features are standardized using training‑time statistics.

---

# 🔐 Future Improvements

* API key authentication
* Rate limiting
* Batch prediction endpoint
* Model versioning
* Monitoring & drift detection
* Feature store integration


# 📜 License

MIT License
