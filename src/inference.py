"""
Customer Churn Prediction - Model Inference & Explainability Module
Loads trained models, computes prediction metrics, risk categorization, and SHAP explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from tensorflow import keras


def load_churn_model(model_paths: list[str] | None = None):
    """Load the trained Keras Sequential Neural Network model."""
    if model_paths is None:
        model_paths = [
            "models/neural_network_final.keras",
            "models/neural_network.keras",
            "models/neural_network_v2.keras",
        ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = keras.models.load_model(path)
                return model, path
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
    raise FileNotFoundError(f"Could not find a valid Keras model in {model_paths}")


def load_baseline_model(path: str = "models/logistic_regression_baseline.pkl"):
    """Load baseline Logistic Regression model if available."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Warning: Could not load baseline model: {e}")
    return None


def get_risk_tier(probability: float, threshold: float = 0.35) -> dict:
    """Categorize churn probability into risk tiers with badge metadata."""
    pct = probability * 100
    if pct < 20.0:
        return {
            "tier": "Low Risk",
            "color": "#10B981",  # Emerald Green
            "badge_class": "badge-low",
            "summary": "Customer displays high loyalty indicators. Low churn intervention needed.",
        }
    elif pct < threshold * 100:
        return {
            "tier": "Moderate Risk (Below Decision Threshold)",
            "color": "#F59E0B",  # Amber
            "badge_class": "badge-moderate",
            "summary": "Customer exhibits some churn signals but remains below the action threshold.",
        }
    elif pct < 60.0:
        return {
            "tier": "High Risk (Action Recommended)",
            "color": "#F97316",  # Orange
            "badge_class": "badge-high",
            "summary": "Customer exceeds the 35% recall-optimized threshold. Proactive retention offer suggested.",
        }
    else:
        return {
            "tier": "Critical Risk (Immediate Intervention)",
            "color": "#EF4444",  # Red
            "badge_class": "badge-critical",
            "summary": "Customer is extremely likely to churn. Urgent retention campaign required.",
        }


def predict_churn(
    model,
    processed_df: pd.DataFrame,
    threshold: float = 0.35,
    baseline_model=None,
) -> dict:
    """Computes neural network and baseline predictions."""
    X_input = processed_df.values.astype("float32")
    raw_proba = float(model.predict(X_input, verbose=0).ravel()[0])

    is_churn = bool(raw_proba >= threshold)
    risk_info = get_risk_tier(raw_proba, threshold)

    result = {
        "probability": raw_proba,
        "probability_pct": raw_proba * 100,
        "threshold": threshold,
        "is_churn": is_churn,
        "prediction_label": "Will Churn" if is_churn else "Will Stay (Retained)",
        "risk_tier": risk_info["tier"],
        "risk_color": risk_info["color"],
        "risk_summary": risk_info["summary"],
    }

    if baseline_model is not None:
        try:
            base_proba = float(baseline_model.predict_proba(X_input)[:, 1][0])
            result["baseline_probability"] = base_proba
            result["baseline_probability_pct"] = base_proba * 100
            result["baseline_is_churn"] = bool(base_proba >= 0.5)
        except Exception:
            result["baseline_probability"] = None

    return result


def compute_shap_contributions(
    model,
    processed_df: pd.DataFrame,
    feature_names: list[str],
    background_samples: np.ndarray | None = None,
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Computes SHAP feature importance for a single customer prediction.
    Returns a clean DataFrame of top contributing features (positive & negative impact).
    """
    if background_samples is None:
        if os.path.exists("data/X_train.npy"):
            X_train = np.load("data/X_train.npy", allow_pickle=True).astype("float32")
            background_samples = X_train[:50]
        else:
            background_samples = np.zeros((10, len(feature_names)), dtype="float32")

    def model_predict_fn(x):
        return model.predict(x, verbose=0).ravel()

    explainer = shap.Explainer(model_predict_fn, background_samples)
    shap_explanation = explainer(processed_df.values)
    shap_values = shap_explanation.values[0]

    feature_impact = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP_Value": shap_values,
            "Abs_Impact": np.abs(shap_values),
            "Direction": np.where(
                shap_values > 0, "Increases Churn Risk", "Decreases Churn Risk"
            ),
        }
    )

    feature_impact = feature_impact.sort_values(
        by="Abs_Impact", ascending=False
    ).head(top_k)
    return feature_impact
