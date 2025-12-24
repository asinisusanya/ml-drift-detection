import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib

# -----------------------
# Load cleaned data
# -----------------------
df = pd.read_csv("data/clean_credit_data.csv")

TARGET = "default_risk"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# -----------------------
# Identify feature types
# -----------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical features:", categorical_cols)
print("Numeric features:", numeric_cols)

# -----------------------
# Train-validation split
# Reference data = training set
# -----------------------
X_ref, X_val, y_ref, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------
# Preprocessing
# -----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# -----------------------
# Model
# -----------------------
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# -----------------------
# Train model
# -----------------------
pipeline.fit(X_ref, y_ref)

# -----------------------
# Evaluate baseline performance
# -----------------------
y_val_pred = pipeline.predict(X_val)
y_val_prob = pipeline.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_val_pred)
auc = roc_auc_score(y_val, y_val_prob)

print(f"Baseline Accuracy: {acc:.4f}")
print(f"Baseline ROC-AUC:  {auc:.4f}")

# -----------------------
# Save model & reference data
# -----------------------
joblib.dump(pipeline, "model.pkl")
X_ref.to_csv("data/reference_data.csv", index=False)

print("Model and reference data saved.")
