# Model Drift Detection in Credit Risk Prediction

An empirical study analyzing the impact of data drift and concept drift on a credit risk prediction model using statistical drift detection, performance evaluation, and explainable AI.

---

## 📌 Project Overview

Machine learning models deployed in real-world decision systems often face evolving data distributions.  
This project demonstrates how such changes—referred to as *model drift*—can silently degrade model performance if not monitored.

The study focuses on:
- Simulating realistic data drift and concept drift scenarios
- Detecting drift using statistical methods
- Evaluating predictive performance under increasing drift severity
- Explaining model failure using SHAP-based feature attribution

The goal is **diagnosis and analysis**, not mitigation, emphasizing the importance of drift monitoring in reliable ML systems.

---

## 🧱 Project Structure
```
ml-drift-detection/
│
├── data/
│ ├── clean_credit_data.csv
│ ├── reference_data.csv
│ ├── current_mild_drift.csv
│ ├── current_moderate_drift.csv
│ └── current_severe_drift.csv
│
├── graphs/
│ ├── Figure_1.png
│ ├── Figure_2.png
│ ├── Figure_3.png
│ └── Figure_4.png
│
├── step1_prepare_data.py
├── step2_train_model.py
├── step3_simulate_drift.py
├── step4_detect_drift.py
├── step5_evaluate_performance.py
├── step6_visualize_results.py
├── step7_shap_explainability.py
│
├── model.pkl
├── drift_report.csv
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **German Credit Risk Dataset**
- Source: https://www.kaggle.com/datasets/uciml/german-credit

A synthetic binary target variable was constructed using domain-inspired rules to enable controlled drift analysis.

---

## ⚙️ Methodology Pipeline
```
Reference Data
↓
Train Model
↓
Simulate Drift
↓
Detect Drift
↓
Evaluate Performance
↓
Explain Failure
```

---

## 🔍 Drift Detection Methods

- **Kolmogorov–Smirnov Test (KS Test)**  
  Detects statistical distribution changes in numerical features

- **Population Stability Index (PSI)**  
  Measures population-level shifts between reference and current data

---

## 📈 Results Summary

- Model performance remains stable under mild and moderate data drift
- Severe concept drift causes a sharp collapse in accuracy and ROC-AUC
- Statistical drift detection does not always imply immediate performance degradation
- SHAP analysis highlights features most sensitive to drift-induced failure

---

## 🧠 Explainability

SHAP (SHapley Additive exPlanations) is used to:
- Identify key predictive features
- Explain model behavior under drift
- Diagnose failure under concept drift scenarios

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

python step1_prepare_data.py
python step2_train_model.py
python step3_simulate_drift.py
python step4_detect_drift.py
python step5_evaluate_performance.py
python step6_visualize_results.py
python step7_shap_explainability.py
```
## 👩‍💻 Author

Asini Susanya Karunarathna
