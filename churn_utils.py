"""Shared helpers for the Customer Churn Prediction project."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_addons", "is_new_customer",
]

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean TotalCharges and create two engineered features.

    - num_addons: how many add-on services the customer has (0-6)
    - is_new_customer: 1 if tenure is 12 months or less
    """
    df = df.copy()
    # In the raw Telco file TotalCharges has blank strings for brand-new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    return df


def build_pipeline(model) -> Pipeline:
    """Feature engineering -> imputing/scaling/encoding -> model, in one object.

    Keeping everything in one Pipeline means the exact same steps run in
    cross-validation, on the test set, and inside the Streamlit app, so there is
    no data leakage and no mismatch between training and prediction.
    """
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", preprocess),
        ("model", model),
    ])
