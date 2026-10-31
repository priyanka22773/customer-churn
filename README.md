# Customer Churn Prediction

Predicts which telecom customers are likely to leave (churn) using the Telco Customer Churn dataset, so a business can target retention offers.

## Problem
Fill in: why churn matters and what the model is meant to help with.

## Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 20 features (demographics, services, contract, billing) and a Yes/No churn target. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in the project folder (not included in this repo).

## Approach
1. Cleaning: dropped `customerID`, converted `TotalCharges` to numeric, imputed missing values.
2. EDA: churn rate by contract, internet service, payment method; tenure and monthly charges vs churn.
3. Feature engineering: `num_addons`, `is_new_customer`.
4. Class imbalance: stratified split, `class_weight="balanced"`, evaluated with recall, F1 and ROC-AUC rather than accuracy alone.
5. Models compared with 5-fold cross-validation: Logistic Regression, KNN, Random Forest, Gradient Boosting.
6. Best model tuned with GridSearchCV (optimising F1) and evaluated on a held-out test set.
7. Permutation importance to explain which features drive churn.
8. Streamlit app for single-customer predictions.

## Results
Fill in with your own numbers after running the notebook.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| KNN | | | | | |
| Random Forest | | | | | |
| Gradient Boosting | | | | | |

Best model: ...  Test-set F1: ...  ROC-AUC: ...

Key findings: ...

## Run it
```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb   # run all cells; saves churn_model.joblib
streamlit run app.py
```

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Project structure
- `churn_prediction.ipynb`: full analysis and modelling
- `churn_utils.py`: feature engineering and preprocessing pipeline
- `app.py`: Streamlit app
- `requirements.txt`
