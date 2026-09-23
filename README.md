# Customer Churn Prediction (Keras/TensorFlow)

An end-to-end machine learning and deep learning project to predict customer churn for a telecommunications provider using an Artificial Neural Network (ANN) built with Keras/TensorFlow, benchmarked against a Logistic Regression baseline model, and explained via SHAP interpretability.

---

## Project Status

**In Progress** — **Phases 1, 2, and 3 completed** (Data Understanding, Exploratory Data Analysis, and Preprocessing & Feature Engineering). Next up: Baseline Modeling (Phase 4).

| Phase | Description | Status |
|---|---|:---:|
| **Phase 1** | Data Understanding & Data Cleaning | Completed |
| **Phase 2** | Exploratory Data Analysis (EDA) | Completed |
| **Phase 3** | Preprocessing, Feature Engineering & Splitting | Completed |
| **Phase 4** | Baseline Model (Logistic Regression) | Next |
| **Phase 5** | Deep Learning Model (Keras/TensorFlow ANN) | Planned |
| **Phase 6** | Hyperparameter Tuning & Regularization | Planned |
| **Phase 7** | Model Evaluation & Diagnostics | Planned |
| **Phase 8** | Model Comparison (Baseline vs. Neural Network) | Planned |
| **Phase 9** | Model Interpretability (SHAP / Feature Importance) | Planned |
| **Phase 10** | Interactive Streamlit Deployment | Planned |

---

## Problem Statement

Customer churn is one of the most critical business metrics for telecommunication companies. Acquiring a new customer can cost 5 to 25 times more than retaining an existing one. Identifying high-risk customers early enables proactive retention strategies, personalized incentives, and optimized customer service interventions.

This project develops an end-to-end predictive pipeline to:
1. Identify high-risk churn customers accurately from account, demographic, and usage features.
2. Compare a traditional linear baseline (Logistic Regression) against a Deep Neural Network (Keras/TensorFlow).
3. Provide model transparency and explainability using SHAP values to uncover key churn drivers for business decision-makers.
4. Deploy an interactive Streamlit web application for real-time customer churn probability scoring.

---

## Dataset Overview

- **Source:** [IBM Telco Customer Churn Dataset (Kaggle / IBM Sample Data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Observations:** 7,043 customer records
- **Raw Features:** 33 columns (demographics, subscribed services, account/billing details, geographic info, and churn flags)
- **Target Variable:** `Churn Value` (`1` = Churned, `0` = Retained)

### Data Cleaning & Column Decisions (Phase 1)

During data auditing in [`01_understanding _the_dataset.ipynb`](notebooks/01_understanding%20_the_dataset.ipynb), columns were categorized and cleaned:

| Category | Columns | Action | Rationale |
|---|---|:---:|---|
| **Identifiers & Constants** | `CustomerID`, `Count`, `Country`, `State` | **Dropped** | `CustomerID` carries no predictive signal; `Country` (all US) and `State` (all CA) have zero variance. |
| **Redundant Geography** | `Lat Long`, `Latitude`, `Longitude`, `Zip Code` | **Dropped** | Redundant spatial information. Regional patterns are captured via `City`. |
| **Target Leakage / Duplicates** | `Churn Label`, `Churn Score`, `Churn Reason` | **Dropped** | `Churn Label` is a duplicate of `Churn Value`. `Churn Score` is a pre-existing model output unavailable in production. `Churn Reason` is only populated after churn occurs (1,869 non-null). |
| **Legitimate Features** | `CLTV` (Customer Lifetime Value) | **Kept** | Calculable for active customers prior to churn; shows valid distributional relationship with churn. |
| **Service & Account Features** | Demographics, Services, Contract, Payment, Charges | **Kept** | 17 core demographic, billing, and subscription features retained. |

#### Missing Value Resolution
- `Total Charges`: 11 records contained blank whitespace strings (`" "`), all corresponding to new customers with `Tenure Months = 0`. These were coerced to numeric float and imputed with `0.0`.
- Cleaned dataset exported to [`data/telco_churn_clean.xlsx`](data/telco_churn_clean.xlsx) (7,043 rows × 22 columns).

---

## Key Findings from Exploratory Data Analysis (Phase 2)

EDA conducted in [`02_eda.ipynb`](notebooks/02_eda.ipynb) revealed key drivers of customer churn:

1. **Target Class Imbalance:**
   - **Retained (0):** 73.46% (5,174 customers)
   - **Churned (1):** 26.54% (1,869 customers)
   - *Implication:* Accuracy alone is an inadequate metric; evaluation will focus on ROC-AUC, PR-AUC, Precision, Recall, and F1-Score.

2. **Customer Tenure:**
   - Churned customers have a median tenure of **10 months** compared to **38 months** for retained customers (correlation: **-0.352**).
   - The highest churn hazard occurs within the first 1–6 months of onboarding.

3. **Contract Type:**
   - **Month-to-month contracts:** **42.71%** churn rate.
   - **One-year contracts:** **11.27%** churn rate.
   - **Two-year contracts:** **2.83%** churn rate.
   - *Implication:* Contract length is among the strongest single predictors of customer retention.

4. **Monthly Charges:**
   - Churned customers pay a higher mean monthly charge (**$74.44**) compared to retained customers (**$61.27**) (correlation: **+0.193**).

5. **Internet Service Type:**
   - **Fiber optic:** **41.89%** churn rate (highest among internet types, likely linked to higher prices or service issues).
   - **DSL:** **18.96%** churn rate.
   - **No Internet:** **7.41%** churn rate.

6. **Payment Method:**
   - **Electronic check:** **45.29%** churn rate.
   - **Mailed check:** **19.11%** churn rate.
   - **Bank transfer (automatic):** **16.71%** churn rate.
   - **Credit card (automatic):** **15.24%** churn rate.

---

## Preprocessing & Feature Engineering (Phase 3)

Implemented in [`03_preprocessing.ipynb`](notebooks/03_preprocessing.ipynb):

1. **Binary Encoding:**
   - Strict binary features (`Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Paperless Billing`) mapped to `0`/`1`.
2. **Multi-Categorical Encoding:**
   - One-hot encoding with `drop_first=True` applied to `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, and `Payment Method`.
3. **High-Cardinality Handling (`City`):**
   - Retained the top 10 most frequent cities (`Los Angeles`, `San Diego`, `San Jose`, `Sacramento`, `San Francisco`, `Fresno`, `Long Beach`, `Oakland`, `Stockton`, `Glendale`) and grouped all remaining 1,119 cities into `'Other'`, followed by one-hot encoding (`drop_first=True`).
4. **Stratified Train / Validation / Test Splitting:**
   - **Train Set (70%):** 4,929 samples
   - **Validation Set (15%):** 1,057 samples
   - **Test Set (15%):** 1,057 samples
   - Stratification on `Churn Value` ensures identical class proportions (~26.5% positive) across all three subsets.
5. **Feature Scaling (Leakage Prevention):**
   - `StandardScaler` fitted **strictly on `X_train`** numerical features (`Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`) and applied to transform `X_val` and `X_test`.
6. **Final Feature Matrix:** 41 input features prepared for modeling.

---

## Project Structure

```
customer-churn-prediction-keras/
├── data/
│   ├── Telco_customer_churn.xlsx   # Raw IBM dataset
│   └── telco_churn_clean.xlsx      # Cleaned dataset (post-Phase 1)
├── notebooks/
│   ├── 01_understanding _the_dataset.ipynb  # Phase 1: Data audit, column decisions & cleaning
│   ├── 02_eda.ipynb                         # Phase 2: Exploratory data analysis & statistical insights
│   └── 03_preprocessing.ipynb               # Phase 3: Encoding, scaling & stratified splitting
├── src/                            # Modular Python modules (data, pipeline, train, eval)
├── models/                         # Saved trained models (.keras, .pkl scalers/encoders)
├── app/                            # Streamlit web application
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

---

## How to Run

### 1. Clone Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/RuchithaAV/customer-churn-prediction-keras.git
cd customer-churn-prediction-keras

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks Sequentially

1. **`notebooks/01_understanding _the_dataset.ipynb`** — Data inspection and initial cleaning.
2. **`notebooks/02_eda.ipynb`** — Exploratory data analysis and visualizations.
3. **`notebooks/03_preprocessing.ipynb`** — Feature engineering, encoding, scaling, and train/val/test splits.

---

## Tech Stack & Dependencies

- **Language:** Python 3.11+
- **Data Manipulation:** `pandas`, `numpy`, `openpyxl`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn`
- **Deep Learning:** `tensorflow`, `keras`
- **Explainability:** `shap`
- **Deployment:** `streamlit`

---

## Author

**Ruchitha Vithana**


