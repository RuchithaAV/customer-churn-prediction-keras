# Customer Churn Prediction (Keras/TensorFlow)

End-to-end customer churn prediction using a Keras/TensorFlow neural network, benchmarked against a Logistic Regression baseline.

## Status

🚧 **In progress** — currently on Phase 1 (data understanding).

## Problem Statement

_TODO: 1–2 sentences on why churn prediction matters for a telecom business, and what this project sets out to do. Fill in after Phase 1._

## Dataset

- **Source:** [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,043 customers, 21 columns
- **Target variable:** `Churn` (Yes/No)
- _TODO: Why this dataset is suitable — fill in after Phase 1._

## Methodology

_TODO: Brief pipeline summary (EDA → preprocessing → baseline → neural network → evaluation → interpretability → deployment). Expand as each phase completes._

## Exploratory Data Analysis (EDA)

_TODO: Key findings — churn distribution, notable relationships between features and churn, visualizations. Fill in after Phase 2._

## Model Architecture

_TODO: Logistic Regression baseline description + Keras neural network architecture (layers, activations, loss, optimizer) with justification for each choice. Fill in after Phases 4–5._

## Results

_TODO: Training/validation curves, confusion matrix, precision/recall/F1, ROC-AUC. Fill in after Phase 7._

## Model Comparison

_TODO: Logistic Regression vs. Neural Network — performance differences and trade-offs, not just a winner. Fill in after Phase 8._

## Interpretability

_TODO: Feature importance / SHAP findings — which customer characteristics are associated with higher churn risk. Fill in after Phase 9._

## Project Structure

```
project/
├── data/           # raw and processed datasets
├── notebooks/       # exploratory and development notebooks
├── src/              # reusable preprocessing/training/evaluation code
├── models/          # saved trained models
├── app/                # Streamlit prediction app
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

_TODO: Setup and run instructions — fill in once the pipeline is runnable end-to-end._

```bash
git clone <repo-url>
cd customer-churn-prediction-keras
pip install -r requirements.txt
```

## Limitations

## Future Improvements

_TODO: Ideas for extending the project — e.g. hyperparameter tuning, additional models, richer deployment._

## Author

Ruchitha Vithana
