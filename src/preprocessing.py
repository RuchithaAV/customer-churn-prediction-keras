"""
Customer Churn Prediction - Preprocessing Pipeline
Replicates exact feature engineering, dummy encoding, reindexing, and scaling from training.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

TOP_10_CITIES = [
    "Los Angeles",
    "San Diego",
    "San Jose",
    "Sacramento",
    "San Francisco",
    "Fresno",
    "Long Beach",
    "Oakland",
    "Stockton",
    "Glendale",
]

NUMERICAL_COLS = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]

BINARY_COLS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
]

MULTI_CAT_COLS = [
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method",
]


def load_feature_names(filepath: str = "data/feature_names.json") -> list[str]:
    """Load exact feature column order expected by the neural network."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scaler(filepath: str = "models/scaler.pkl"):
    """Load fitted StandardScaler from training."""
    return joblib.load(filepath)


def load_threshold(filepath: str = "models/final_threshold.txt", default: float = 0.35) -> float:
    """Load the decision threshold selected for optimal recall."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except Exception:
            return default
    return default


def preprocess_customer_input(
    raw_input: dict,
    feature_names: list[str],
    scaler,
) -> pd.DataFrame:
    """
    Transforms single customer raw dictionary into the exact 41-dimensional
    scaled and one-hot encoded vector expected by Keras.
    """
    df = pd.DataFrame([raw_input])

    # 1. Binary mappings (Yes/No -> 1/0, Male/Female -> 1/0)
    binary_mappings = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0,
        1: 1,
        0: 0,
    }
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(binary_mappings).fillna(0).astype(int)

    # 2. City bucketing (Top 10 California cities + 'Other')
    if "City" in df.columns:
        city_val = df["City"].iloc[0]
        if city_val not in TOP_10_CITIES:
            df["City"] = "Other"

    # 3. Multi-category one-hot encoding (without drop_first here because we reindex)
    all_multi_cats = MULTI_CAT_COLS + (["City"] if "City" in df.columns else [])
    df_encoded = pd.get_dummies(df, columns=all_multi_cats, drop_first=False, dtype=int)

    # 4. Reindex onto the exact 41 training feature columns, filling missing dummies with 0
    df_reindexed = df_encoded.reindex(columns=feature_names, fill_value=0)

    # 5. Apply fitted StandardScaler to the 4 numerical columns
    for num_col in NUMERICAL_COLS:
        if num_col not in df_reindexed.columns:
            df_reindexed[num_col] = 0.0
        df_reindexed[num_col] = pd.to_numeric(df_reindexed[num_col], errors="coerce").fillna(0.0)

    df_reindexed[NUMERICAL_COLS] = scaler.transform(df_reindexed[NUMERICAL_COLS])

    return df_reindexed.astype("float32")


def get_sample_customer_presets() -> dict[str, dict]:
    """Provides sample profiles for easy UI testing."""
    return {
        "🔴 High Risk Churner (Month-to-month, Fiber, Electronic check, Short tenure)": {
            "Gender": "Female",
            "Senior Citizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "Tenure Months": 2,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charges": 70.70,
            "Total Charges": 151.65,
            "CLTV": 2701,
            "City": "Los Angeles",
        },
        "🟢 Low Risk Loyal Customer (Two year contract, DSL, Automatic payment, Long tenure)": {
            "Gender": "Male",
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure Months": 65,
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "Yes",
            "Device Protection": "Yes",
            "Tech Support": "Yes",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Two year",
            "Paperless Billing": "No",
            "Payment Method": "Bank transfer (automatic)",
            "Monthly Charges": 85.50,
            "Total Charges": 5500.00,
            "CLTV": 5400,
            "City": "San Francisco",
        },
        "🟡 Borderline / Moderate Risk Customer (1 Year Contract, Mixed Services)": {
            "Gender": "Female",
            "Senior Citizen": "Yes",
            "Partner": "No",
            "Dependents": "No",
            "Tenure Months": 18,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "Fiber optic",
            "Online Security": "Yes",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "No",
            "Contract": "One year",
            "Paperless Billing": "Yes",
            "Payment Method": "Credit card (automatic)",
            "Monthly Charges": 79.80,
            "Total Charges": 1436.40,
            "CLTV": 4100,
            "City": "San Diego",
        },
    }
