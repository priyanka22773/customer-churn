"""Streamlit app: predict churn probability for a single customer.

Run the notebook first (it saves churn_model.joblib), then:
    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

import churn_utils  # noqa: F401  (needed so joblib can find add_features)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")
st.caption("Enter customer details to estimate the probability that they will churn.")


@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


model = load_model()

yes_no = ["Yes", "No"]
addon_opts = ["Yes", "No", "No internet service"]

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", yes_no)
    dependents = st.selectbox("Dependents", yes_no)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", yes_no)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=1.0)
    total = st.number_input("Total charges", 0.0, 10000.0, float(tenure) * 70.0, step=10.0)
with col2:
    phone = st.selectbox("Phone service", yes_no)
    multiple = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online security", addon_opts)
    backup = st.selectbox("Online backup", addon_opts)
    protection = st.selectbox("Device protection", addon_opts)
    support = st.selectbox("Tech support", addon_opts)
    tv = st.selectbox("Streaming TV", addon_opts)
    movies = st.selectbox("Streaming movies", addon_opts)

if st.button("Predict churn"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    prob = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{prob:.1%}")
    if prob >= 0.5:
        st.error("High risk: this customer is likely to churn.")
    elif prob >= 0.3:
        st.warning("Medium risk: consider a retention offer.")
    else:
        st.success("Low risk: this customer is likely to stay.")
