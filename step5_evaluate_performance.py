import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------
# Load model
# -----------------------
pipeline = joblib.load("model.pkl")

# -----------------------
# Load datasets
# -----------------------
ref = pd.read_csv("data/reference_data.csv")
mild = pd.read_csv("data/current_mild_drift.csv")
moderate = pd.read_csv("data/current_moderate_drift.csv")
severe = pd.read_csv("data/current_severe_drift.csv")

datasets = {
    "reference": ref,
    "mild": mild,
    "moderate": moderate,
    "severe": severe
}

# -----------------------
# Training-time target logic
# -----------------------
def training_target(df):
    y = (
        (df["Credit amount"] > 5000).astype(int) +
        (df["Duration"] > 36).astype(int) +
        (df["Saving accounts"].isin(["none", "little"])).astype(int)
    )
    return (y >= 2).astype(int)

# -----------------------
# Evaluate performance
# -----------------------
results = []

for name, df in datasets.items():

    X = df.copy()

    # Use DIFFERENT target for severe (concept drift)
    if name == "severe" and "default_risk" in df.columns:
        y_true = df["default_risk"]
        X = X.drop(columns=["default_risk"])
    else:
        y_true = training_target(df)

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    results.append({
        "dataset": name,
        "accuracy": acc,
        "roc_auc": auc
    })

results_df = pd.DataFrame(results)
print(results_df)
