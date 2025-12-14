import random
import csv
from datetime import datetime, timedelta

NUM_TRANSACTIONS = 50000
OUTPUT_FILE = "data/transactions_50k.csv"

users = [f"U{str(i).zfill(5)}" for i in range(2000)]
merchants = [f"M{str(i).zfill(4)}" for i in range(300)]
countries = ["IN", "US", "UK", "SG", "AE"]
devices = ["mobile", "web", "pos"]

start_time = datetime(2024, 1, 1)

def generate_transaction(i):
    user = random.choice(users)
    merchant = random.choice(merchants)
    amount = round(random.expovariate(1/500), 2)  # skewed amounts
    country = random.choice(countries)
    device = random.choice(devices)
    timestamp = start_time + timedelta(seconds=random.randint(0, 86400 * 30))

    fraud_prob = 0.01
    if amount > 3000:
        fraud_prob += 0.10
    if merchant.endswith("99"):
        fraud_prob += 0.05

    is_fraud = 1 if random.random() < fraud_prob else 0

    return [
        i,
        user,
        merchant,
        amount,
        country,
        device,
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        is_fraud
    ]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "transaction_id",
        "user_id",
        "merchant_id",
        "amount",
        "country",
        "device_type",
        "timestamp",
        "is_fraud"
    ])

    for i in range(1, NUM_TRANSACTIONS + 1):
        writer.writerow(generate_transaction(i))

print(f"✅ Generated {NUM_TRANSACTIONS} transactions → {OUTPUT_FILE}")
