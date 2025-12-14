from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

def create_spark_session():
    return SparkSession.builder \
        .appName("FraudDataIngestion") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

def ingest_data(input_path):
    spark = create_spark_session()

    print("🔵 Reading data from:", input_path)

    df = spark.read.csv(input_path, header=True, inferSchema=True)

    print("📌 Sample Rows:")
    df.show(5)

    print("📌 Schema:")
    df.printSchema()

    return df

if __name__ == "__main__":
    INPUT_FILE = "data/transactions_50k.csv"
    ingest_data(INPUT_FILE)
