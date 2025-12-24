import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Load trained pipeline
# -----------------------
pipeline = joblib.load("model.pkl")

# -----------------------
# Load reference data
# -----------------------
X_ref = pd.read_csv("data/reference_data.csv")

# -----------------------
# Extract preprocessing + model
# -----------------------
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

# Transform data
X_transformed = preprocessor.transform(X_ref)

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# -----------------------
# SHAP Explainer
# -----------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# -----------------------
# SHAP Summary Plot (BAR)
# -----------------------
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=True
)
