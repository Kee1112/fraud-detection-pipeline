#build_features
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, avg, when, log
)
import os


def create_spark_session():
    return SparkSession.builder \
        .appName("FraudFeatureEngineering") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .getOrCreate()


def build_features(input_path):
    spark = create_spark_session()

    # ✅ Parquet read (correct)
    df = spark.read.parquet(input_path)

    # ✅ FIX: timestamp → epoch seconds
    df = df.withColumn(
        "event_time",
        col("timestamp").cast("long")
    )

    # User velocity features (1 hour)
    user_window_1h = Window.partitionBy("user_id") \
        .orderBy(col("event_time")) \
        .rangeBetween(-3600, 0)

    df = df.withColumn(
        "tx_count_user_1h",
        count("*").over(user_window_1h)
    )

    # User spending behavior (24 hours)
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
        avg("is_fraud").alias("merchant_fraud_rate"),
        avg("amount").alias("merchant_avg_amount")
    )

    df = df.join(merchant_stats, on="merchant_id", how="left")

    # Transaction-level features
    df = df.withColumn("amount_log", log(col("amount") + 1))
    df = df.withColumn(
        "is_high_amount",
        when(col("amount") > 3000, 1).otherwise(0)
    )

    # Amount vs user's recent behavior
    df = df.withColumn(
        "amount_vs_user_avg_24h",
        col("amount") / (col("avg_amount_user_24h") + 1
    )
    )
    # Amount vs merchant behavior
    df = df.withColumn(
        "amount_vs_merchant_avg",
        col("amount") / (col("merchant_avg_amount") + 1)
    
    )
    merchant_window_1h = Window.partitionBy("merchant_id") \
        .orderBy(col("event_time")) \
        .rangeBetween(-3600, 0)

    df = df.withColumn(
        "tx_count_merchant_1h",
        count("*").over(merchant_window_1h)
    )

    df = df.withColumn(
        "merchant_velocity_spike",
        when(col("tx_count_merchant_1h") > 20, 1).otherwise(0)
    )
    df = df.withColumn(
        "high_amount_mobile",
        when(
            (col("is_high_amount") == 1) & (col("device_type") == "mobile"),
            1
    ).otherwise(0)
    )
    df = df.withColumn(
        "foreign_high_amount",
        when(
            (col("country") == "OTHER") & (col("amount") > 2500),
            1
    ).otherwise(0)
   )

    df = df.withColumn(
    "repeat_user_fast",
    when(col("tx_count_user_1h") > 3, 1).otherwise(0)
   )
    df = df.withColumn(
    "merchant_risky_high_amount",
    when(
      (col("merchant_fraud_rate") > 0.04) & (col("amount" > 3000), 1)
      .otherwise(0)

    )
   )
    


    return df


if __name__ == "__main__":
    INPUT_FILE = "fraud-detection-pipeline/data/transactions_big"
    OUTPUT_FILE = "fraud-detection-pipeline/data/transactions_features"

    df_features = build_features(INPUT_FILE)

    # ⚠️ Minor improvement: output path consistency
    os.makedirs("fraud-detection-pipeline/data", exist_ok=True)

    df_features \
        .write \
        .mode("overwrite") \
        .parquet(OUTPUT_FILE)

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
