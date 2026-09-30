"""
Customer Churn Prediction - Interactive Web Application
Built with Streamlit and TensorFlow/Keras for Deep Learning Deployment.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.inference import (
    compute_shap_contributions,
    load_baseline_model,
    load_churn_model,
    predict_churn,
)
from src.preprocessing import (
    TOP_10_CITIES,
    get_sample_customer_presets,
    load_feature_names,
    load_scaler,
    load_threshold,
    preprocess_customer_input,
)

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor | Keras Neural Network",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern UI
st.markdown(
    """
    <style>
    /* Global Styles & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    /* Risk Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        text-align: center;
    }
    .badge-churn {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }
    .badge-retain {
        background-color: #DCFCE7;
        color: #16A34A;
        border: 1px solid #86EFAC;
    }
    
    /* Section Cards */
    .section-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_and_artifacts():
    """Load and cache models and preprocessing artifacts."""
    model, model_path = load_churn_model()
    baseline = load_baseline_model()
    scaler = load_scaler()
    features = load_feature_names()
    threshold = load_threshold()

    # Load background data for SHAP if available
    background = None
    if os.path.exists("data/X_train.npy"):
        try:
            X_train = np.load("data/X_train.npy", allow_pickle=True).astype("float32")
            background = X_train[:50]
        except Exception:
            background = None

    return model, model_path, baseline, scaler, features, threshold, background


# Load Artifacts
try:
    (
        model,
        model_path,
        baseline_model,
        scaler,
        feature_names,
        decision_threshold,
        shap_background,
    ) = get_model_and_artifacts()
except Exception as e:
    st.error(f" Error loading model artifacts: {e}")
    st.stop()


# Top Hero Header
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title"> Customer Churn Prediction Engine</div>
        <div class="hero-subtitle">
            Trained on Telco Customer Churn using a <b>Keras Deep Neural Network</b> (Sequential Dense 32 &rarr; 16 &rarr; 1).<br>
            Optimized with a <b>Decision Threshold of {decision_threshold:.2f}</b> to prioritize churn recall over precision.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Sidebar Configuration & Preset Selection
with st.sidebar:
    st.markdown("### Input Controls & Presets")

    presets = get_sample_customer_presets()
    selected_preset = st.selectbox(
        "Load Customer Profile Preset:",
        options=["Custom Customer"] + list(presets.keys()),
        index=1,
        help="Quickly populate the form with realistic customer archetypes.",
    )

    preset_data = {}
    if selected_preset in presets:
        preset_data = presets[selected_preset]

    st.markdown("---")
    st.markdown("###  Model Parameters")
    custom_threshold = st.slider(
        "Decision Threshold:",
        min_value=0.10,
        max_value=0.90,
        value=float(decision_threshold),
        step=0.01,
        help="Default 0.35 chosen to catch more churners (higher recall).",
    )

    show_shap = st.checkbox("Show SHAP Explanation", value=True)
    show_baseline = st.checkbox("Compare with Baseline Logistic Reg.", value=True)
    show_vector = st.checkbox("Inspect 41-dim Feature Vector", value=False)

    st.markdown("---")
    st.caption(" **Artifact Details:**")
    st.caption(f"• Model: `{os.path.basename(model_path)}`")
    st.caption("• Features: 41 One-Hot Encoded Columns")
    st.caption("• Scaler: StandardScaler (Fitted)")


# Main Input Form
st.markdown("###  Enter Customer Profile")

with st.form("churn_prediction_form"):
    tab1, tab2, tab3 = st.tabs(
        [" Demographics & Location", " Services & Add-ons", " Contract & Financials"]
    )

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                index=0 if preset_data.get("Gender", "Male") == "Male" else 1,
            )
            senior_citizen = st.selectbox(
                "Senior Citizen",
                options=["No", "Yes"],
                index=0 if preset_data.get("Senior Citizen", "No") == "No" else 1,
            )
        with col2:
            partner = st.selectbox(
                "Partner",
                options=["No", "Yes"],
                index=0 if preset_data.get("Partner", "No") == "No" else 1,
            )
            dependents = st.selectbox(
                "Dependents",
                options=["No", "Yes"],
                index=0 if preset_data.get("Dependents", "No") == "No" else 1,
            )
        with col3:
            city_options = TOP_10_CITIES + ["Other"]
            default_city = preset_data.get("City", "Los Angeles")
            city_idx = (
                city_options.index(default_city)
                if default_city in city_options
                else city_options.index("Other")
            )
            city = st.selectbox("California City", options=city_options, index=city_idx)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            phone_service = st.selectbox(
                "Phone Service",
                options=["Yes", "No"],
                index=0 if preset_data.get("Phone Service", "Yes") == "Yes" else 1,
            )
            mult_opts = ["No", "Yes", "No phone service"]
            mult_val = preset_data.get("Multiple Lines", "No")
            mult_idx = mult_opts.index(mult_val) if mult_val in mult_opts else 0
            multiple_lines = st.selectbox(
                "Multiple Lines", options=mult_opts, index=mult_idx
            )

            inet_opts = ["DSL", "Fiber optic", "No"]
            inet_val = preset_data.get("Internet Service", "DSL")
            inet_idx = inet_opts.index(inet_val) if inet_val in inet_opts else 0
            internet_service = st.selectbox(
                "Internet Service", options=inet_opts, index=inet_idx
            )

        with col2:
            sec_opts = ["No", "Yes", "No internet service"]
            sec_val = preset_data.get("Online Security", "No")
            sec_idx = sec_opts.index(sec_val) if sec_val in sec_opts else 0
            online_security = st.selectbox(
                "Online Security", options=sec_opts, index=sec_idx
            )

            bk_opts = ["No", "Yes", "No internet service"]
            bk_val = preset_data.get("Online Backup", "No")
            bk_idx = bk_opts.index(bk_val) if bk_val in bk_opts else 0
            online_backup = st.selectbox(
                "Online Backup", options=bk_opts, index=bk_idx
            )

            dev_opts = ["No", "Yes", "No internet service"]
            dev_val = preset_data.get("Device Protection", "No")
            dev_idx = dev_opts.index(dev_val) if dev_val in dev_opts else 0
            device_protection = st.selectbox(
                "Device Protection", options=dev_opts, index=dev_idx
            )

        with col3:
            tech_opts = ["No", "Yes", "No internet service"]
            tech_val = preset_data.get("Tech Support", "No")
            tech_idx = tech_opts.index(tech_val) if tech_val in tech_opts else 0
            tech_support = st.selectbox(
                "Tech Support", options=tech_opts, index=tech_idx
            )

            tv_opts = ["No", "Yes", "No internet service"]
            tv_val = preset_data.get("Streaming TV", "No")
            tv_idx = tv_opts.index(tv_val) if tv_val in tv_opts else 0
            streaming_tv = st.selectbox(
                "Streaming TV", options=tv_opts, index=tv_idx
            )

            mov_opts = ["No", "Yes", "No internet service"]
            mov_val = preset_data.get("Streaming Movies", "No")
            mov_idx = mov_opts.index(mov_val) if mov_val in mov_opts else 0
            streaming_movies = st.selectbox(
                "Streaming Movies", options=mov_opts, index=mov_idx
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            contract_opts = ["Month-to-month", "One year", "Two year"]
            cont_val = preset_data.get("Contract", "Month-to-month")
            cont_idx = contract_opts.index(cont_val) if cont_val in contract_opts else 0
            contract = st.selectbox("Contract Type", options=contract_opts, index=cont_idx)

            paperless = st.selectbox(
                "Paperless Billing",
                options=["Yes", "No"],
                index=0 if preset_data.get("Paperless Billing", "Yes") == "Yes" else 1,
            )

            pay_opts = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            pay_val = preset_data.get("Payment Method", "Electronic check")
            pay_idx = pay_opts.index(pay_val) if pay_val in pay_opts else 0
            payment_method = st.selectbox(
                "Payment Method", options=pay_opts, index=pay_idx
            )

        with col2:
            tenure_months = st.slider(
                "Tenure Months (Customer Longevity)",
                min_value=0,
                max_value=72,
                value=int(preset_data.get("Tenure Months", 12)),
                help="Number of months customer has stayed with the company.",
            )

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=150.0,
                value=float(preset_data.get("Monthly Charges", 65.0)),
                step=0.5,
            )

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(
                    preset_data.get(
                        "Total Charges", max(monthly_charges * max(tenure_months, 1), 50.0)
                    )
                ),
                step=10.0,
            )

            cltv = st.number_input(
                "Customer Lifetime Value (CLTV Score)",
                min_value=1000,
                max_value=7000,
                value=int(preset_data.get("CLTV", 4000)),
                step=50,
            )

    submit_button = st.form_submit_button(
        "🔮 Run Churn Prediction", use_container_width=True, type="primary"
    )


# Prediction Execution & Results
raw_customer_dict = {
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure_months,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
    "CLTV": cltv,
    "City": city,
}

# Preprocess single customer input
try:
    processed_vector = preprocess_customer_input(
        raw_customer_dict, feature_names, scaler
    )
    prediction = predict_churn(
        model,
        processed_vector,
        threshold=custom_threshold,
        baseline_model=baseline_model if show_baseline else None,
    )
except Exception as e:
    st.error(f" Preprocessing / Inference Error: {e}")
    st.stop()

st.markdown("---")
st.markdown("##  Prediction Results & Risk Assessment")

# Display Result Cards
res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

with res_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">
                Neural Network Churn Probability
            </div>
            <div style="font-size: 2.5rem; font-weight: 700; color: {prediction['risk_color']}; margin: 0.4rem 0;">
                {prediction['probability_pct']:.1f}%
            </div>
            <div style="font-size: 0.9rem; color: #475569;">
                Decision Threshold: <b>{prediction['threshold'] * 100:.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_col2:
    badge_class = "badge-churn" if prediction["is_churn"] else "badge-retain"
    st.markdown(
        f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">
                Classification Decision
            </div>
            <div style="margin: 0.8rem 0;">
                <span class="badge-pill {badge_class}" style="font-size: 1.1rem; padding: 0.5rem 1.2rem;">
                    {'⚠️ ' + prediction['prediction_label'] if prediction['is_churn'] else '✅ ' + prediction['prediction_label']}
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #64748B;">
                {prediction['risk_tier']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_col3:
    if show_baseline and baseline_model is not None and prediction.get("baseline_probability") is not None:
        base_proba = prediction["baseline_probability_pct"]
        base_label = "Churn" if prediction["baseline_is_churn"] else "Retain"
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">
                    Baseline Logistic Reg.
                </div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #475569; margin: 0.4rem 0;">
                    {base_proba:.1f}%
                </div>
                <div style="font-size: 0.9rem; color: #64748B;">
                    Default 50% cutoff: <b>{base_label}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">
                    Retention Priority
                </div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #1E293B; margin-top: 0.75rem;">
                    {prediction['risk_summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# SHAP Feature Contribution Chart
if show_shap:
    st.markdown("###  Key Churn Drivers (SHAP Local Explanation)")
    with st.spinner("Calculating SHAP feature attribution..."):
        try:
            shap_df = compute_shap_contributions(
                model,
                processed_vector,
                feature_names,
                background_samples=shap_background,
                top_k=8,
            )

            fig, ax = plt.subplots(figsize=(10, 4.5))
            colors = [
                "#EF4444" if val > 0 else "#10B981" for val in shap_df["SHAP_Value"]
            ]

            y_positions = np.arange(len(shap_df))
            bars = ax.barh(
                y_positions,
                shap_df["SHAP_Value"],
                color=colors,
                height=0.6,
                edgecolor="none",
            )
            ax.set_yticks(y_positions)
            ax.set_yticklabels(shap_df["Feature"], fontsize=10, fontweight="500")
            ax.invert_yaxis()  # Top feature at top
            ax.axvline(0, color="#94A3B8", linewidth=0.8, linestyle="--")

            # Value labels on bars
            for bar in bars:
                width = bar.get_width()
                offset = 0.003 if width >= 0 else -0.003
                ha = "left" if width >= 0 else "right"
                ax.text(
                    width + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:+.3f}",
                    va="center",
                    ha=ha,
                    fontsize=9,
                    color="#1E293B",
                    fontweight="600",
                )

            ax.set_xlabel("SHAP Value (Impact on Churn Probability)", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CBD5E1")
            ax.spines["bottom"].set_color("#CBD5E1")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "🔴 **Red bars** increase churn risk for this customer. 🟢 **Green bars** decrease churn risk (encourage retention)."
            )
        except Exception as e:
            st.warning(f"Could not compute SHAP plot: {e}")


# Feature Inspection Table
if show_vector:
    st.markdown("### 🔬 41-Dimensional Preprocessed Vector Inspection")
    non_zero_cols = [
        col for col in feature_names if processed_vector[col].iloc[0] != 0
    ]
    st.write(
        f"**Active (Non-Zero) Features for this Customer ({len(non_zero_cols)} of 41):**"
    )

    vector_display = []
    for col in feature_names:
        val = processed_vector[col].iloc[0]
        vector_display.append(
            {
                "Feature Name": col,
                "Processed Value": f"{val:.4f}",
                "Is Active": "✅ Yes" if val != 0 else "—",
            }
        )

    st.dataframe(pd.DataFrame(vector_display), use_container_width=True, height=300)
