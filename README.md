# Customer Churn Prediction (Keras/TensorFlow)

An end-to-end machine learning and deep learning project to predict customer churn for a telecommunications provider using an Artificial Neural Network (ANN) built with Keras/TensorFlow, benchmarked against a Logistic Regression baseline model, regularized with Dropout & L2 penalties, evaluated via comprehensive diagnostics & threshold tuning, and explained via SHAP interpretability.

---

## Project Status

**In Progress** — **Phases 1 through 8 completed** (Data Understanding, Exploratory Data Analysis, Preprocessing & Feature Engineering, Baseline Modeling, Deep Learning ANN Modeling, Hyperparameter Tuning & Regularization, Full Model Evaluation & Diagnostics, and Model Comparison). **Phase 9 (Model Interpretability with SHAP)** is next.

| Phase | Description | Status |
|---|---|:---:|
| **Phase 1** | Data Understanding & Data Cleaning | Completed |
| **Phase 2** | Exploratory Data Analysis (EDA) | Completed |
| **Phase 3** | Preprocessing, Feature Engineering & Splitting | Completed |
| **Phase 4** | Baseline Model (Logistic Regression) | Completed |
| **Phase 5** | Deep Learning Model (Keras/TensorFlow ANN) | Completed |
| **Phase 6** | Hyperparameter Tuning & Regularization | Completed |
| **Phase 7** | Full Model Evaluation & Diagnostics | Completed |
| **Phase 8** | Model Comparison (Baseline vs. Neural Networks) | Completed |
| **Phase 9** | Model Interpretability (SHAP / Feature Importance) | Planned |
| **Phase 10** | Interactive Streamlit Deployment | Planned |

---

## Problem Statement

Customer churn is one of the most critical business metrics for telecommunication companies. Acquiring a new customer can cost 5 to 25 times more than retaining an existing one. Identifying high-risk customers early enables proactive retention strategies, personalized incentives, and optimized customer service interventions.

This project develops an end-to-end predictive pipeline to:
1. Identify high-risk churn customers accurately from account, demographic, and usage features.
2. Compare a traditional linear baseline (Logistic Regression) against deep neural architectures (Keras/TensorFlow).
3. Mitigate overfitting using L2 regularization and Dropout, optimizing decision thresholds to align with business retention economics.
4. Provide model transparency and explainability using SHAP values to uncover key churn drivers for business decision-makers.
5. Deploy an interactive Streamlit web application for real-time customer churn probability scoring.

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
   - *Implication:* Accuracy alone is an inadequate metric; evaluation focuses on ROC-AUC, PR-AUC, Precision, Recall, and F1-Score.

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
   - **Train Set (70%):** 4,929 samples (`data/X_train.npy`, `data/y_train.npy`)
   - **Validation Set (15%):** 1,057 samples (`data/X_val.npy`, `data/y_val.npy`)
   - **Test Set (15%):** 1,057 samples (`data/X_test.npy`, `data/y_test.npy`)
   - Stratification on `Churn Value` ensures identical class proportions (~26.5% positive) across all three subsets.
5. **Feature Scaling (Leakage Prevention):**
   - `StandardScaler` fitted **strictly on `X_train`** numerical features (`Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`) and applied to transform `X_val` and `X_test`.
6. **Final Feature Matrix:** 41 input features prepared and tracked via [`data/feature_names.json`](data/feature_names.json).

---

## Baseline Modeling — Logistic Regression (Phase 4)

Implemented in [`04_Baseline_model.ipynb`](notebooks/04_Baseline_model.ipynb):

A linear baseline model was established to benchmark future deep learning architectures against a simple, interpretable statistical standard.

- **Model:** `LogisticRegression(max_iter=1000, random_state=42)`
- **Validation Results (1,057 validation samples):**
  - **Overall Accuracy:** **79.75% (~80%)** (vs. 73.46% naive "always predict no churn" majority class baseline — a +6.29% improvement).
  - **Class 0 (Retained):** Precision: `0.84`, Recall: `0.89`, F1-Score: `0.87` (Support: 776).
  - **Class 1 (Churned):** Precision: `0.64`, Recall: `0.55`, F1-Score: `0.59` (Support: 281).
  - **Macro Average:** Precision: `0.74`, Recall: `0.72`, F1-Score: `0.73`.
  - **Weighted Average:** Precision: `0.79`, Recall: `0.80`, F1-Score: `0.79`.
  - **ROC-AUC:** `0.8435`.
- **Key Takeaway:** The baseline captures 55% of actual churners (Recall = 0.55) with 64% precision.
- **Saved Model Artifact:** [`models/logistic_regression_baseline.pkl`](models/logistic_regression_baseline.pkl)

---

## Deep Learning Model — Initial Keras/TensorFlow ANN (Phase 5)

Implemented in [`05_Deep_Learning_Model.ipynb`](notebooks/05_Deep_Learning_Model.ipynb):

A multi-layer feedforward Artificial Neural Network (ANN) was constructed to model complex non-linear feature interactions.

### 1. Architecture Overview (Model v1)

```
Input (41 features)
       │
Dense Layer 1 (32 units, ReLU activation) ─── [1,344 params]
       │
Dense Layer 2 (16 units, ReLU activation) ─── [528 params]
       │
Output Layer (1 unit, Sigmoid activation) ─── [17 params]
──────────────────────────────────────────────────────────
Total Trainable Parameters: 1,889
```

### 2. Compilation & Training Dynamics
- **Optimizer:** `Adam` | **Loss:** `binary_crossentropy` | **Batch Size:** `32` | **Max Epochs:** `50`
- **Overfitting Control:** `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
- **Training Trajectory:** EarlyStopping triggered early at **Epoch 10**, restoring best weights from **Epoch 5** (`val_loss: 0.4195`, `val_accuracy: 79.85%`).
- **Saved Model Artifact:** [`models/neural_network.keras`](models/neural_network.keras)

---

## Hyperparameter Tuning & Regularization (Phase 6)

Implemented in [`06_hyperparameter_tuning.ipynb`](notebooks/06_hyperparameter_tuning.ipynb):

To address early plateauing and overfitting observed in the initial ANN, regularization techniques and threshold tuning were applied.

### 1. Regularized Architecture (Model v2)

```
Input (41 features)
       │
Dense Layer 1 (32 units, ReLU, L2=0.001) ───── [1,344 params]
       │
Dropout Layer 1 (rate = 0.3)
       │
Dense Layer 2 (16 units, ReLU, L2=0.001) ───── [528 params]
       │
Dropout Layer 2 (rate = 0.3)
       │
Output Layer (1 unit, Sigmoid) ─────────────── [17 params]
──────────────────────────────────────────────────────────
Total Trainable Parameters: 1,889
```

### 2. Regularization Impact
- Adding **L2 weight regularization (0.001)** and **Dropout (0.3)** significantly altered learning dynamics: validation loss continued to improve stably through **Epoch 31** (compared to Epoch 10 in v1).
- Overfitting was effectively mitigated, allowing deeper gradient optimization and boosting discriminative performance (ROC-AUC: `0.8441` → `0.8469`).

### 3. Decision Threshold Optimization
- In customer retention, the business cost of a **False Negative** (losing a churned customer and their entire CLTV) far outweighs that of a **False Positive** (a proactive discount or outreach call to a retained customer).
- Tuning the classification threshold from the default `0.50` to **`0.35`** boosted Churn Recall from **0.56 to 0.70**, ensuring **70% of churners** are proactively captured.
- **Saved Artifacts:**
  - Regularized Model: [`models/neural_network_v2.keras`](models/neural_network_v2.keras)
  - Optimal Threshold: [`models/final_threshold.txt`](models/final_threshold.txt) (`0.35`)

---

## Full Model Evaluation & Diagnostics (Phase 7)

Implemented in [`07_Full_evaluation.ipynb`](notebooks/07_Full_evaluation.ipynb):

Comprehensive evaluation and diagnostic analysis were performed on the 1,057 validation records:

1. **Confusion Matrix Analysis:**
   - Evaluated true positives, false positives, true negatives, and false negatives across models and thresholds.
2. **ROC Curve & AUC Score:**
   - Both Logistic Regression (`0.8435`) and the Neural Networks (`0.8441` for v1, `0.8469` for v2) show strong discriminative capability.
3. **Threshold Sensitivity & Trade-offs:**
   - Shifting threshold from 0.50 to 0.35 creates an intentional, business-justified trade-off: Churn Recall increases significantly (53% → 69-70%) with a manageable trade-off in precision (65% → 55-58%).

---

## Model Comparison & Benchmark Summary (Phase 8)

Comprehensive performance comparison across all tested configurations on the validation set (1,057 samples):

| Model Configuration | Decision Threshold | Accuracy | Churn Precision | Churn Recall | Churn F1-Score | ROC-AUC | Overfitting Control | Artifact File |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **Naive Baseline** | — | 73.46% | 0.00 | 0.00 | 0.00 | — | None (predicts majority) | — |
| **Logistic Regression** | 0.50 | 79.75% | 0.64 | 0.55 | 0.59 | 0.8435 | L2 penalty (C=1.0) | [`models/logistic_regression_baseline.pkl`](models/logistic_regression_baseline.pkl) |
| **Initial ANN (Model v1)** | 0.50 | 79.85% | 0.65 | 0.53 | 0.58 | 0.8441 | EarlyStopping (patience=5) | [`models/neural_network.keras`](models/neural_network.keras) |
| **Initial ANN (Model v1)** | 0.35 | 78.00% | 0.58 | 0.69 | 0.63 | 0.8441 | EarlyStopping (patience=5) | [`models/neural_network.keras`](models/neural_network.keras) |
| **Regularized ANN (Model v2)** | 0.50 | 80.89% | **0.67** | 0.56 | 0.61 | **0.8469** | L2 (0.001) + Dropout (0.3) | [`models/neural_network_v2.keras`](models/neural_network_v2.keras) |
| **Regularized ANN (Model v2) ⭐** | **0.35** | **77.39%** | **0.55** | **0.70** | **0.62** | **0.8469** | **L2 (0.001) + Dropout (0.3)** | **[`models/neural_network_v2.keras`](models/neural_network_v2.keras)** |

> ⭐ **Selected Final Configuration:** `Regularized ANN (Model v2) @ 0.35 Threshold` — achieves the highest churn recall (**0.70**) and highest ROC-AUC (**0.8469**), directly maximizing customer retention potential.

---

## Project Structure

```
customer-churn-prediction-keras/
├── data/
│   ├── Telco_customer_churn.xlsx         # Raw IBM dataset
│   ├── telco_churn_clean.xlsx            # Cleaned dataset (post-Phase 1)
│   ├── X_train.npy / y_train.npy         # Preprocessed training set (70%, 4929 samples)
│   ├── X_val.npy / y_val.npy             # Preprocessed validation set (15%, 1057 samples)
│   ├── X_test.npy / y_test.npy           # Preprocessed test set (15%, 1057 samples)
│   └── feature_names.json                # 41 engineered feature column names
├── notebooks/
│   ├── 01_understanding _the_dataset.ipynb # Phase 1: Data audit, column decisions & cleaning
│   ├── 02_eda.ipynb                       # Phase 2: Exploratory data analysis & statistical insights
│   ├── 03_preprocessing.ipynb             # Phase 3: Encoding, scaling & stratified splitting
│   ├── 04_Baseline_model.ipynb            # Phase 4: Logistic Regression baseline model
│   ├── 05_Deep_Learning_Model.ipynb       # Phase 5: Initial Keras ANN architecture & training
│   ├── 06_hyperparameter_tuning.ipynb     # Phase 6: L2 regularization, Dropout, & threshold tuning
│   ├── 07_Full_evaluation.ipynb           # Phase 7: Model evaluation, confusion matrix & diagnostics
│   └── 08_Model_comparison.ipynb          # Phase 8: Comprehensive model benchmarking
├── src/                                  # Modular Python source code
├── models/                               # Saved model artifacts
│   ├── logistic_regression_baseline.pkl   # Trained Logistic Regression model
│   ├── neural_network.keras               # Initial trained Keras ANN model (v1)
│   ├── neural_network_v2.keras            # Regularized Keras ANN model (v2)
│   └── final_threshold.txt                # Selected optimal classification threshold (0.35)
├── app/                                  # Streamlit web application
├── requirements.txt                      # Project dependencies
├── README.md                             # Project documentation
└── .gitignore                            # Git ignore rules
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

1. **`notebooks/01_understanding _the_dataset.ipynb`** — Data inspection, column categorization, and cleaning.
2. **`notebooks/02_eda.ipynb`** — Exploratory data analysis and visualization of churn drivers.
3. **`notebooks/03_preprocessing.ipynb`** — Feature encoding, standard scaling, and stratified train/val/test splitting.
4. **`notebooks/04_Baseline_model.ipynb`** — Train and evaluate the Logistic Regression benchmark model.
5. **`notebooks/05_Deep_Learning_Model.ipynb`** — Build, compile, train, and save the initial Keras ANN.
6. **`notebooks/06_hyperparameter_tuning.ipynb`** — Regularize ANN with Dropout + L2 and perform threshold tuning.
7. **`notebooks/07_Full_evaluation.ipynb`** — Detailed evaluation diagnostics, ROC curves, confusion matrices, and trade-off analysis.
8. **`notebooks/08_Model_comparison.ipynb`** — Side-by-side performance benchmarking and test set evaluation.

---

## Tech Stack & Dependencies

- **Language:** Python 3.11+
- **Data Manipulation:** `pandas`, `numpy`, `openpyxl`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn`, `joblib`
- **Deep Learning:** `tensorflow`, `keras`
- **Explainability:** `shap`
- **Deployment:** `streamlit`

---

## Author

**Ruchitha Vithana**


