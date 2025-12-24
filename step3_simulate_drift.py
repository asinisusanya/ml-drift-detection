import pandas as pd
import numpy as np

# -----------------------
# Load reference data
# -----------------------
ref = pd.read_csv("data/reference_data.csv")

# Create copies
mild = ref.copy()
moderate = ref.copy()
severe = ref.copy()

np.random.seed(42)

# -----------------------
# 1. NUMERIC DATA DRIFT
# -----------------------
numeric_cols = ["Credit amount", "Duration", "Age"]

for col in numeric_cols:
    # Mild drift
    mild[col] = mild[col] * np.random.normal(1.05, 0.02, len(mild))

    # Moderate drift
    moderate[col] = moderate[col] * np.random.normal(1.15, 0.05, len(moderate))

    # Severe drift
    severe[col] = severe[col] * np.random.normal(1.30, 0.10, len(severe))

# -----------------------
# 2. CATEGORICAL DATA DRIFT
# -----------------------
def shift_categories(df, column, from_vals, to_val, frac):
    """
    Randomly shift a fraction of rows from from_vals to to_val
    """
    mask = df[column].isin(from_vals)
    idx = df[mask].sample(frac=frac, random_state=42).index
    df.loc[idx, column] = to_val

# Savings behavior deteriorates
shift_categories(mild, "Saving accounts", ["rich", "moderate"], "little", 0.10)
shift_categories(moderate, "Saving accounts", ["rich", "moderate"], "little", 0.25)
shift_categories(severe, "Saving accounts", ["rich", "moderate"], "none", 0.50)

# Checking account behavior deteriorates
shift_categories(moderate, "Checking account", ["moderate", "rich"], "little", 0.20)
shift_categories(severe, "Checking account", ["moderate", "rich"], "none", 0.40)

# -----------------------
# 3. CONCEPT DRIFT (SEVERE ONLY)
# -----------------------
# Policy / economic regime change:
# Default logic is no longer the same as training

severe["default_risk"] = (
    (severe["Credit amount"] < 4000).astype(int) +
    (severe["Duration"] > 48).astype(int)
)

severe["default_risk"] = (severe["default_risk"] >= 1).astype(int)

# -----------------------
# Save datasets
# -----------------------
mild.to_csv("data/current_mild_drift.csv", index=False)
moderate.to_csv("data/current_moderate_drift.csv", index=False)
severe.to_csv("data/current_severe_drift.csv", index=False)

print("Step 3 completed:")
print("- Mild data drift created")
print("- Moderate data drift created")
print("- Severe data + concept drift created")
