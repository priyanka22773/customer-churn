# Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to leave (churn), built with Python and Scikit-learn. It covers data cleaning, EDA, feature engineering, class-imbalance handling, model comparison with cross-validation, hyperparameter tuning, and a Streamlit app for single-customer predictions.

**Author:** Priyanka Ghulannavar | [GitHub](https://github.com/priyanka22773) | [[LinkedIn](www.linkedin.com/in/priyanka-ghulannavar-63930926b)]

## Problem
Winning a new customer usually costs more than keeping an existing one. If a telecom company can identify customers who are likely to leave, it can offer them targeted retention deals before they go. This project trains a classifier to predict churn from customer demographics, services, contract and billing data.

## Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers and 21 columns, including a Yes/No `Churn` target. About 26.5% of customers churned, so the classes are imbalanced.

The dataset is not stored in this repo. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in the project folder.

## Approach
1. **Cleaning:** dropped `customerID`, converted `TotalCharges` to numeric (blank for new customers), and imputed missing values with the median.
2. **EDA:** churn rate by contract type, internet service and payment method; tenure and monthly charges versus churn.
3. **Feature engineering:** `num_addons` (count of add-on services) and `is_new_customer` (tenure of 12 months or less).
4. **Class imbalance:** stratified train/test split and `class_weight="balanced"`. Models are judged on recall, F1 and ROC-AUC, because accuracy alone is misleading when about 73% of customers stay.
5. **Model comparison:** Logistic Regression, KNN, Random Forest and Gradient Boosting, compared with 5-fold cross-validation inside a single Scikit-learn `Pipeline`, so the same preprocessing runs in training, testing and the app.
6. **Tuning:** `GridSearchCV` on the best model, optimising F1.
7. **Evaluation:** classification report, confusion matrix, ROC curve and a decision-threshold comparison on a held-out test set.
8. **Explainability:** permutation importance to show which features drive predictions.
9. **App:** a Streamlit app that returns a churn probability for one customer.

## Results

Cross-validation results (5-fold, training set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.750 | 0.519 | 0.799 | 0.629 | 0.846 |
| KNN | 0.788 | 0.610 | 0.555 | 0.581 | 0.825 |
| Random Forest | 0.787 | 0.634 | 0.472 | 0.541 | 0.823 |
| Gradient Boosting | 0.803 | 0.661 | 0.531 | 0.589 | 0.848 |

**Best model:** Gradient Boosting (highest cross-validated ROC-AUC), tuned with GridSearchCV to `learning_rate=0.1`, `max_depth=2`, `n_estimators=200` (best CV F1: 0.59).

**Test-set performance (1,409 customers, churn class):** Precision 0.67 | Recall 0.52 | F1 0.58 | ROC-AUC 0.844 | Accuracy 0.80 (versus 0.735 for always predicting "stayed").

**Key findings:**
- New customers (tenure of 12 months or less) churned at 47.4%, compared with 17.1% for longer-tenured customers.
- Permutation importance ranked `tenure` first, then `Contract`, then `InternetService`.
- Lowering the decision threshold from 0.5 to 0.3 raised recall from 0.52 to 0.76 but cut precision from 0.67 to 0.53. F1 was highest at a threshold of 0.4 (0.634), so the best threshold depends on the cost of missing a churner versus contacting someone who would have stayed.
- Logistic Regression had the highest recall (0.80) and F1 (0.63) in cross-validation, while Gradient Boosting had the best ROC-AUC and accuracy. The model choice therefore depends on whether the goal is to catch more churners or to rank customers by risk.

## Limitations and next steps
- Results come from one dataset and one train/test split.
- The Telco data has no time dimension, so it cannot show how churn changes over time.
- Next steps: try SMOTE or XGBoost, calibrate predicted probabilities, and test on newer data.

## How to run
```bash
git clone https://github.com/priyanka22773/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```
1. Download the dataset from Kaggle and put the CSV in this folder.
2. Open `churn_prediction.ipynb` and run all cells. This saves `churn_model.joblib`.
3. Start the app: `streamlit run app.py`

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Project structure
- `churn_prediction.ipynb`: full analysis and modelling
- `churn_utils.py`: feature engineering and preprocessing pipeline
- `app.py`: Streamlit prediction app
- `churn_model.joblib`: trained model
- `requirements.txt`: dependencies
