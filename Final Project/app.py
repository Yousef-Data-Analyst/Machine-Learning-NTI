import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telco Customer Churn Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load("telco_churn_pipeline.pkl")


artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]
threshold = artifacts["threshold"]
feature_columns = artifacts["feature_columns"]

st.title("Telco Customer Churn Prediction System")
st.markdown("Predict customer churn probability and retention risk using the tuned **HistGradientBoosting** model.")

# Helper to get classes safely from saved LabelEncoders (with fallback if column was pre-encoded)
def get_options(col, fallback):
    if col in encoders:
        return list(encoders[col].classes_)
    return fallback

st.subheader("1. Customer Profile & Account Details")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", get_options("Gender", ["Male", "Female"]))
    senior = st.selectbox("Senior Citizen", get_options("Senior Citizen", ["No", "Yes"]))
    partner = st.selectbox("Partner", get_options("Partner", ["No", "Yes"]))
    dependents = st.selectbox("Dependents", get_options("Dependents", ["No", "Yes"]))
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

with col2:
    contract = st.selectbox("Contract", get_options("Contract", ["Month-to-month", "One year", "Two year"]))
    paperless = st.selectbox("Paperless Billing", get_options("Paperless Billing", ["Yes", "No"]))
    payment = st.selectbox(
        "Payment Method",
        get_options("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0, value=float(tenure * monthly_charges), step=10.0)
    cltv = st.number_input("Customer Lifetime Value (CLTV)", min_value=1000, max_value=8000, value=4000, step=50)

with col3:
    phone = st.selectbox("Phone Service", get_options("Phone Service", ["Yes", "No"]))
    multiple_lines = st.selectbox("Multiple Lines", get_options("Multiple Lines", ["No", "Yes", "No phone service"]))
    internet = st.selectbox("Internet Service", get_options("Internet Service", ["Fiber optic", "DSL", "No"]))
    online_sec = st.selectbox("Online Security", get_options("Online Security", ["No", "Yes", "No internet service"]))
    online_backup = st.selectbox("Online Backup", get_options("Online Backup", ["No", "Yes", "No internet service"]))
    device_prot = st.selectbox("Device Protection", get_options("Device Protection", ["No", "Yes", "No internet service"]))
    tech_support = st.selectbox("Tech Support", get_options("Tech Support", ["No", "Yes", "No internet service"]))
    streaming_tv = st.selectbox("Streaming TV", get_options("Streaming TV", ["No", "Yes", "No internet service"]))
    streaming_movies = st.selectbox("Streaming Movies", get_options("Streaming Movies", ["No", "Yes", "No internet service"]))

# Build raw input dictionary matching your dataset's exact column names
raw_input = {
    "Gender": gender,
    "Senior Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Phone Service": phone,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet,
    "Online Security": online_sec,
    "Online Backup": online_backup,
    "Device Protection": device_prot,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless,
    "Payment Method": payment,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
    "CLTV": cltv,
}

st.divider()

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    # Keep only the exact columns used during training and enforce identical column order
    input_df = pd.DataFrame([{col: raw_input[col] for col in feature_columns}])

    # Apply the exact saved LabelEncoders
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Apply the exact saved StandardScaler
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=feature_columns
    )

    # Predict probability & apply optimal threshold
    churn_prob = float(model.predict_proba(input_scaled)[0, 1])
    prediction = int(churn_prob >= threshold)

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
    res_col2.metric("Decision Threshold", f"{threshold * 100:.1f}%")
    res_col3.metric("Predicted Status", "HIGH RISK (Churn)" if prediction == 1 else "LOW RISK (Retain)")

    st.progress(min(max(churn_prob, 0.0), 1.0))

    if prediction == 1:
        st.error(
            f"**Action Needed:** This customer has a **{churn_prob * 100:.1f}%** probability of churning "
            f"(above the {threshold * 100:.1f}% risk threshold). Consider offering a 1-year contract discount, "
            f"bundled Tech Support/Online Security, or automatic payment incentives."
        )
    else:
        st.success(
            f"**Retained:** This customer has a **{churn_prob * 100:.1f}%** probability of churning "
            f"(below the {threshold * 100:.1f}% risk threshold)."
        )