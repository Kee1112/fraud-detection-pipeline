from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

def create_spark_session():
    return SparkSession.builder \
        .appName("FraudModelTraining") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

def load_features(path):
    spark = create_spark_session()
    df = spark.read.csv(path, header=True, inferSchema=True)
    return df

if __name__ == "__main__":

    FEATURES_FILE = "data/transactions_50k.csv"

    spark = create_spark_session()
    df = spark.read.csv(FEATURES_FILE, header=True, inferSchema=True)

    # Basic features for first model
    feature_cols = [
        "amount",
        "is_high_amount",
        "is_fraud"
    ]

    pdf = df.select(*feature_cols).dropna().toPandas()

    X = pdf[["amount", "is_high_amount"]]
    y = pdf["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("fraud_detection_experiment")

    with mlflow.start_run():
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("features", "amount,is_high_amount")
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, "fraud_model")

        print("✅ Model trained")
        print("ROC AUC:", auc)
