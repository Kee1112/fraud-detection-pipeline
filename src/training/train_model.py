from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def create_spark_session():
    return SparkSession.builder \
        .appName("FraudModelTraining") \
        .getOrCreate()


def train():
    spark = create_spark_session()

    # 1️⃣ Load feature-engineered data
    df = spark.read.parquet(
        "fraud-detection-pipeline/data/transactions_features"
    )

    # 2️⃣ Feature list (final, strong set)
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

    # 3️⃣ Handle missing values
    df_clean = df.fillna({
        "tx_count_user_1h": 0,
        "amount_vs_user_avg_24h": 0,
        "avg_amount_user_24h": 0,
        "amount_vs_merchant_avg": 0,
        "tx_count_merchant_1h": 0,
        "merchant_velocity_spike": 0,
        "merchant_fraud_rate": 0,
        "amount_log": 0,
        "is_high_amount": 0,
        "foreign_high_amount": 0,
        "high_amount_mobile": 0,
        "repeat_user_fast": 0,
        "merchant_risky_high_amount": 0
    })

    # 4️⃣ Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features"
    )

    df_assembled = assembler.transform(df_clean)

    # 5️⃣ Scale features
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    df_scaled = scaler.fit(df_assembled).transform(df_assembled)

    # 6️⃣ Handle class imbalance
    fraud_ratio = df_scaled.filter(col("is_fraud") == 1).count() / df_scaled.count()

    df_ml = df_scaled.withColumn(
        "label", col("is_fraud")
    ).withColumn(
        "classWeight",
        when(col("label") == 1, 1.0 / fraud_ratio).otherwise(1.0)
    ).select(
        "features",
        "label",
        "classWeight"
    ).cache()

    # 7️⃣ Train / test split
    train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)

    # ======================
    # Logistic Regression
    # ======================
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxIter=30,
        regParam=0.01
    )

    lr_model = lr.fit(train_df)
    lr_preds = lr_model.transform(test_df)

    # ======================
    # Gradient Boosted Trees
    # ======================
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=15,
        maxDepth=5,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=42
    )

    gbt_model = gbt.fit(train_df)
    gbt_preds = gbt_model.transform(test_df)

    # ======================
    # Evaluation
    # ======================
    roc_eval = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderROC"
    )

    pr_eval = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderPR"
    )

    print("📊 Logistic Regression")
    print("ROC AUC:", roc_eval.evaluate(lr_preds))
    print("PR  AUC:", pr_eval.evaluate(lr_preds))

    print("\n📊 Gradient Boosted Trees")
    print("ROC AUC:", roc_eval.evaluate(gbt_preds))
    print("PR  AUC:", pr_eval.evaluate(gbt_preds))

    # 8️⃣ Save best model (GBT)
    gbt_model.save("fraud-detection-pipeline/models/gbt_fraud_model")
    lr_model.save("fraud-detection-pipeline/models/lr_fraud_model")

    print("✅ Models saved successfully")


if __name__ == "__main__":
    train()
