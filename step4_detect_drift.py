import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# -----------------------
# Load datasets
# -----------------------
ref = pd.read_csv("data/reference_data.csv")

mild = pd.read_csv("data/current_mild_drift.csv")
moderate = pd.read_csv("data/current_moderate_drift.csv")
severe = pd.read_csv("data/current_severe_drift.csv")

datasets = {
    "mild": mild,
    "moderate": moderate,
    "severe": severe
}

# -----------------------
# Feature groups
# -----------------------
categorical_cols = ref.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = ref.select_dtypes(include=["int64", "float64"]).columns.tolist()

# -----------------------
# PSI function
# -----------------------
def calculate_psi(expected, actual, buckets=10):
    expected_percents, _ = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=buckets)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (expected_percents - actual_percents) *
        np.log((expected_percents + 1e-6) / (actual_percents + 1e-6))
    )
    return psi

# -----------------------
# Drift detection
# -----------------------
drift_reports = []

for level, df in datasets.items():
    # Numeric features (KS Test)
    for col in numeric_cols:
        stat, p_value = ks_2samp(ref[col], df[col])
        drift_reports.append({
            "drift_level": level,
            "feature": col,
            "method": "KS Test",
            "statistic": stat,
            "p_value": p_value,
            "drift_detected": p_value < 0.05
        })

    # Categorical features (PSI)
    for col in categorical_cols:
        psi = calculate_psi(ref[col].astype("category").cat.codes,
                            df[col].astype("category").cat.codes)
        drift_reports.append({
            "drift_level": level,
            "feature": col,
            "method": "PSI",
            "statistic": psi,
            "p_value": None,
            "drift_detected": psi > 0.2  # common PSI threshold
        })

# -----------------------
# Save report
# -----------------------
drift_df = pd.DataFrame(drift_reports)
drift_df.to_csv("drift_report.csv", index=False)

print("Drift detection completed.")
print(drift_df.groupby(["drift_level", "method"])["drift_detected"].sum())
