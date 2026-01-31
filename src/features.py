import json
import numpy as np

with open("models_a/scaler.json") as f:
    scaler = json.load(f)

FEATURE_ORDER = scaler["feature_order"]
MEAN = np.array(scaler["mean"])
STD = np.array(scaler["std"])


def build_feature_vector(payload: dict) -> np.ndarray:
    """
    payload: raw transaction JSON
    returns: scaled numpy array (1, n_features)
    """

    raw_features = np.array([
        payload.get("tx_count_user_1h", 0),
        payload.get("amount_vs_user_avg_24h", 0),
        payload.get("avg_amount_user_24h", 0),
        payload.get("amount_vs_merchant_avg", 0),
        payload.get("tx_count_merchant_1h", 0),
        payload.get("merchant_velocity_spike", 0),
        payload.get("merchant_fraud_rate", 0),
        payload.get("amount_log", 0),
        payload.get("is_high_amount", 0),
        payload.get("foreign_high_amount", 0),
        payload.get("high_amount_mobile", 0),
        payload.get("repeat_user_fast", 0),
        payload.get("merchant_risky_high_amount", 0),
    ], dtype=float)

    # Standard scaling
    scaled = (raw_features - MEAN) / STD

    return scaled.reshape(1, -1)

