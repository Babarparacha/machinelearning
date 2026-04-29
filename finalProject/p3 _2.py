# ============================================================
# ELECTRICITY THEFT DETECTION - ADVANCED (ANTI-OVERFITTING)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 1. DATA GENERATION + CSV SAVE
# ============================================================

# np.random.seed(42)
# n_samples = 2500

# df = pd.DataFrame()

# # -------------------------------
# # BASIC CONSUMPTION
# # -------------------------------
# df['avg_consumption'] = np.random.normal(300, 100, n_samples).clip(50)
# df['peak_consumption'] = df['avg_consumption'] + np.random.normal(60, 30, n_samples)
# df['previous_month_consumption'] = df['avg_consumption'] + np.random.normal(0, 50, n_samples)

# # -------------------------------
# # TIME BASED FEATURES
# # -------------------------------
# df['weekly_avg_consumption'] = df['avg_consumption'] + np.random.normal(0, 20, n_samples)
# df['daily_variation'] = np.random.uniform(0, 1, n_samples)
# df['usage_spike'] = np.random.uniform(0, 1, n_samples)
# df['usage_drop_ratio'] = np.random.uniform(0, 1, n_samples)

# # -------------------------------
# # ELECTRICAL FEATURES
# # -------------------------------
# df['voltage'] = np.random.normal(220, 10, n_samples)
# df['current'] = np.random.normal(10, 3, n_samples)
# df['power_factor'] = np.random.uniform(0.7, 1.0, n_samples)
# df['frequency'] = np.random.normal(50, 0.5, n_samples)
# df['voltage_fluctuation'] = np.random.uniform(0, 1, n_samples)

# # -------------------------------
# # BILLING FEATURES
# # -------------------------------
# df['billing_amount'] = df['avg_consumption'] * np.random.uniform(0.1, 0.3, n_samples)
# df['avg_bill_last_3_months'] = df['billing_amount'] + np.random.normal(0, 50, n_samples)
# df['bill_difference'] = df['billing_amount'] - df['avg_bill_last_3_months']
# df['payment_delay_days'] = np.random.randint(0, 30, n_samples)
# df['unpaid_bills_count'] = np.random.randint(0, 5, n_samples)

# # -------------------------------
# # GRID + NEIGHBOR DATA
# # -------------------------------
# df['neighbor_avg_consumption'] = df['avg_consumption'] + np.random.normal(0, 40, n_samples)
# df['deviation_from_area'] = df['avg_consumption'] - df['neighbor_avg_consumption']
# df['line_loss'] = np.random.uniform(0, 0.3, n_samples)
# df['transformer_load'] = np.random.uniform(0.5, 1.5, n_samples)

# # -------------------------------
# # FLAGS
# # -------------------------------
# df['tamper_flag'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# # ============================================================
# # TARGET CREATION
# # ============================================================
# theft_score = (
#     0.25 * df['usage_drop_ratio'] +
#     0.2 * df['tamper_flag'] +
#     0.15 * abs(df['deviation_from_area']) +
#     0.15 * df['line_loss'] +
#     0.1 * (1 - df['power_factor']) +
#     0.15 * df['voltage_fluctuation']
# )

# df['theft'] = (theft_score > 0.6).astype(int)

# # SAVE CSV
# csv_path = r"F:\Python course\machineLearning\finalProject\electricity_theft_data_advanced.csv"
# df.to_csv(csv_path, index=False)
# print("✅ CSV Created")

# ============================================================
# 2. LOAD CSV
# ============================================================
data = pd.read_csv(r"F:\Python course\machineLearning\finalProject\electricity_theft_data_advanced.csv")
# data = pd.read_csv(csv_path)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

data['power'] = data['voltage'] * data['current']
data['consumption_ratio'] = data['peak_consumption'] / data['avg_consumption']

data['consumption_change'] = (
    data['avg_consumption'] - data['previous_month_consumption']
) / data['previous_month_consumption']

data['risk_score'] = (
    data['usage_drop_ratio'] * 0.4 +
    data['tamper_flag'] * 0.6
)

data['suspicious_pattern'] = (
    (data['usage_spike'] > 0.8) &
    (data['usage_drop_ratio'] > 0.7)
).astype(int)

print("✅ Feature Engineering Done")

# ============================================================
# 4. SPLIT
# ============================================================
X = data.drop('theft', axis=1)
y = data['theft']

# ============================================================
# 5. SCALING
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================================================
# 7. MODEL (ANTI-OVERFITTING)
# ============================================================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# 8. CROSS VALIDATION
# ============================================================
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

print("\nCross Validation Accuracy:", cv_scores.mean())

# ============================================================
# 9. EVALUATION
# ============================================================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
plt.figure()
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")
plt.show()

plt.figure()
data['theft'].value_counts().plot(kind='bar')
plt.title("Theft Distribution")
plt.show()
# ============================================================
# 11. PREDICTION
# ============================================================
sample = X.iloc[[0]]
sample_scaled = scaler.transform(sample)

print("\n🔍 Prediction:", model.predict(sample_scaled)[0])
print("\n🔍 Prediction (0=Normal, 1=Theft):", model.predict(sample_scaled)[0])