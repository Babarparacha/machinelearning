# ============================================
# ELECTRICITY THEFT DETECTION ML PIPELINE
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. DATA GENERATION (Pakistan Scenario)
# ============================================

np.random.seed(42)

n_samples = 1500

data = pd.DataFrame({
    "monthly_units": np.random.normal(300, 120, n_samples).clip(50, 1000),
    "bill_amount": np.random.normal(8000, 3000, n_samples).clip(1000, 30000),
    "load_shedding_hours": np.random.randint(0, 12, n_samples),
    "connection_age_years": np.random.randint(1, 30, n_samples),
    "area_type": np.random.choice(["urban", "rural"], n_samples),
    "meter_type": np.random.choice(["smart", "analog"], n_samples),
})

# Introduce theft behavior patterns
data["theft"] = (
    (data["monthly_units"] > 500) &
    (data["bill_amount"] < 6000)
).astype(int)

# Add randomness
noise = np.random.binomial(1, 0.1, n_samples)
data["theft"] = data["theft"] ^ noise

print("\nSample Data:")
print(data.head())

# ============================================
# 2. EDA (Exploratory Data Analysis)
# ============================================

plt.figure()
sns.histplot(data["monthly_units"], kde=True)
plt.title("Monthly Units Distribution")
plt.show()

plt.figure()
sns.boxplot(x="theft", y="bill_amount", data=data)
plt.title("Bill Amount vs Theft")
plt.show()

plt.figure()
sns.countplot(x="area_type", hue="theft", data=data)
plt.title("Area Type vs Theft")
plt.show()

# Correlation heatmap
plt.figure()
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

# Convert categorical to numeric
data = pd.get_dummies(data, columns=["area_type", "meter_type"], drop_first=True)

# Feature: cost per unit
data["cost_per_unit"] = data["bill_amount"] / data["monthly_units"]

# Feature: abnormal usage
data["abnormal_usage"] = (data["monthly_units"] > 600).astype(int)

# ============================================
# 4. SPLIT DATA
# ============================================

X = data.drop("theft", axis=1)
y = data["theft"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 5. PIPELINE (SCALING + MODEL)
# ============================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42
    ))
])

# ============================================
# 6. TRAIN MODEL
# ============================================

pipeline.fit(X_train, y_train)

# ============================================
# 7. PREDICTION
# ============================================

y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================

model = pipeline.named_steps["model"]

importances = model.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feat_imp.head(10))

plt.figure()
sns.barplot(x="importance", y="feature", data=feat_imp.head(10))
plt.title("Top 10 Important Features")
plt.show()

# ============================================
# 9. FRAUD / THEFT RISK SCORING (0–100)
# ============================================

probs = pipeline.predict_proba(X_test)[:, 1]

risk_scores = (probs * 100).astype(int)

results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = y_pred
results["risk_score"] = risk_scores

print("\nSample Risk Scores:")
print(results.head(10))

# ============================================
# 10. RISK CATEGORY CLASSIFICATION
# ============================================

def risk_label(score):
    if score < 30:
        return "Low Risk"
    elif score < 70:
        return "Medium Risk"
    else:
        return "High Risk"

results["risk_label"] = results["risk_score"].apply(risk_label)

print("\nRisk Distribution:")
print(results["risk_label"].value_counts())

plt.figure()
sns.countplot(x="risk_label", data=results)
plt.title("Risk Category Distribution")
plt.show()

# ============================================
# 11. SAVE MODEL
# ============================================

# import joblib
# joblib.dump(pipeline, "electricity_theft_model.pkl")

# print("\nModel saved as electricity_theft_model.pkl")

# ============================================
# END OF PIPELINE
# ============================================