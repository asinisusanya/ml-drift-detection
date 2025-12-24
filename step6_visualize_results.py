import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
ref = pd.read_csv("data/reference_data.csv")
mild = pd.read_csv("data/current_mild_drift.csv")
moderate = pd.read_csv("data/current_moderate_drift.csv")
severe = pd.read_csv("data/current_severe_drift.csv")

# Feature to visualize
feature = "Credit amount"

plt.figure(figsize=(8, 5))
sns.kdeplot(ref[feature], label="Reference", linewidth=2)
sns.kdeplot(mild[feature], label="Mild Drift", linestyle="--")
sns.kdeplot(moderate[feature], label="Moderate Drift", linestyle="-.")
sns.kdeplot(severe[feature], label="Severe Drift", linestyle=":")

plt.title(f"Distribution Shift of {feature}")
plt.xlabel(feature)
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Performance results (from Step 5)
results = pd.DataFrame({
    "Dataset": ["Reference", "Mild", "Moderate", "Severe"],
    "ROC_AUC": [1.00, 0.990984, 0.995404, 0.208697]
})

plt.figure(figsize=(7, 4))
plt.plot(results["Dataset"], results["ROC_AUC"], marker="o")
plt.title("Model Performance under Increasing Drift")
plt.xlabel("Drift Scenario")
plt.ylabel("ROC-AUC")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(results["Dataset"], [1.0, 0.9675, 0.9800, 0.2350], label="Accuracy", marker="o")
plt.plot(results["Dataset"], results["ROC_AUC"], label="ROC-AUC", marker="s")

plt.title("Performance Collapse under Concept Drift")
plt.xlabel("Drift Scenario")
plt.ylabel("Metric Value")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
