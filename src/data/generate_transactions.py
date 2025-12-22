from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, when, col, unix_timestamp, from_unixtime

# --------------------------------------------------
# Spark Session
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("FraudDataGenerator") \
    .getOrCreate()

# --------------------------------------------------
# Config
# --------------------------------------------------
N_ROWS = 5_000_000
DAYS_BACK = 30

# --------------------------------------------------
# Base Transactions
# --------------------------------------------------
df = spark.range(N_ROWS) \
    .withColumn("user_id", (rand() * 100_000).cast("int")) \
    .withColumn("merchant_id", (rand() * 5_000).cast("int")) \
    .withColumn("amount", rand() * 5000) \
    .withColumn(
        "country",
        when(rand() > 0.8, "US")
        .when(rand() > 0.6, "IN")
        .when(rand() > 0.4, "UK")
        .otherwise("OTHER")
    ) \
    .withColumn(
        "device_type",
        when(rand() > 0.7, "mobile").otherwise("web")
    )

# --------------------------------------------------
# Inject Risk Profiles (KEY STEP)
# --------------------------------------------------
df = df \
    .withColumn("is_risky_user", when(rand() < 0.02, 1).otherwise(0)) \
    .withColumn("is_risky_merchant", when(rand() < 0.015, 1).otherwise(0))

# --------------------------------------------------
# Fraud Logic (STRUCTURED, LEARNABLE)
# --------------------------------------------------
df = df.withColumn(
    "is_fraud",
    when(
        (col("is_risky_user") == 1) &
        (col("amount") > 2000) &
        (rand() > 0.4),
        1
    ).when(
        (col("is_risky_merchant") == 1) &
        (col("amount") > 1500) &
        (rand() > 0.5),
        1
    ).when(
        (col("country") == "OTHER") &
        (col("device_type") == "mobile") &
        (col("amount") > 2500),
        1
    ).when(
        (col("amount") > 3000) &
        (rand() > 0.8),
        1
    ).otherwise(0)
)

# --------------------------------------------------
# Timestamp (last 30 days)
# --------------------------------------------------
df = df.withColumn(
    "timestamp",
    from_unixtime(
        unix_timestamp() - (rand() * DAYS_BACK * 24 * 60 * 60).cast("int")
    )
)

# --------------------------------------------------
# Write Output
# --------------------------------------------------
df.write \
    .mode("overwrite") \
    .parquet("data/transactions_big")

print("✅ Fraud dataset generated with realistic patterns")
