import pandas as pd
import numpy as np

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("data/german_credit_data.csv")

# -----------------------
# Drop index column
# -----------------------
df = df.drop(columns=["Unnamed: 0"])

# -----------------------
# Handle missing values
# -----------------------
df["Saving accounts"] = df["Saving accounts"].fillna("none")
df["Checking account"] = df["Checking account"].fillna("none")

# -----------------------
# Create synthetic target (default risk)
# -----------------------
np.random.seed(42)

df["default_risk"] = (
    (df["Credit amount"] > 5000).astype(int) +
    (df["Duration"] > 36).astype(int) +
    (df["Saving accounts"].isin(["none", "little"])).astype(int)
)

# Convert to binary
df["default_risk"] = (df["default_risk"] >= 2).astype(int)

print("Class distribution:")
print(df["default_risk"].value_counts(normalize=True))

# -----------------------
# Save cleaned data
# -----------------------
df.to_csv("data/clean_credit_data.csv", index=False)
