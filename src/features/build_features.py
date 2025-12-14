from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, avg, sum, when,
    log, unix_timestamp
)
import os


def create_spark_session():
    return SparkSession.builder \
        .appName("FraudFeatureEngineering") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()

def build_features(input_path):
    spark = create_spark_session()

    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Convert timestamp
    df = df.withColumn(
        "event_time",
        unix_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
    )

    # User velocity features (1 hour window)
    user_window_1h = Window.partitionBy("user_id") \
        .orderBy(col("event_time")) \
        .rangeBetween(-3600, 0)

    df = df.withColumn(
        "tx_count_user_1h",
        count("*").over(user_window_1h)
    )

    # User spending behavior (24 hour window)
    user_window_24h = Window.partitionBy("user_id") \
        .orderBy(col("event_time")) \
        .rangeBetween(-86400, 0)

    df = df.withColumn(
        "avg_amount_user_24h",
        avg("amount").over(user_window_24h)
    )

    # Merchant risk features
    merchant_stats = df.groupBy("merchant_id").agg(
        count("*").alias("tx_count_merchant"),
        avg("is_fraud").alias("merchant_fraud_rate")
    )

    df = df.join(merchant_stats, on="merchant_id", how="left")

    # Transaction-level features
    df = df.withColumn("amount_log", log(col("amount") + 1))
    df = df.withColumn(
        "is_high_amount",
        when(col("amount") > 3000, 1).otherwise(0)
    )

    return df

if __name__ == "__main__":
    INPUT_FILE = "data/transactions_50k.csv"
    OUTPUT_FILE = "data/transactions_features"

    df_features = build_features(INPUT_FILE)

    os.makedirs("data", exist_ok=True)

    df_features.coalesce(1)\
    .write \
    .mode("overwrite") \
    .option("header","true")\
    .csv(OUTPUT_FILE)

    print(f"✅ Features saved to {OUTPUT_FILE}")

    df_features.select(
        "user_id",
        "merchant_id",
        "amount",
        "tx_count_user_1h",
        "avg_amount_user_24h",
        "merchant_fraud_rate",
        "is_high_amount",
        "is_fraud"
    ).show(5)
