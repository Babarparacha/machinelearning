# ============================================================
# ELECTRICITY THEFT DETECTION - FINAL CLEAN VERSION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from scipy.stats import zscore
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. DATA GENERATION + SAVE CSV
# ============================================================

# np.random.seed(42)
# n_samples = 2500

# df = pd.DataFrame()

# df['avg_consumption'] = np.random.normal(300, 100, n_samples).clip(50)
# df['peak_consumption'] = df['avg_consumption'] + np.random.normal(60, 30, n_samples)
# df['previous_month_consumption'] = df['avg_consumption'] + np.random.normal(0, 50, n_samples)

# df['weekly_avg_consumption'] = df['avg_consumption'] + np.random.normal(0, 20, n_samples)
# df['daily_variation'] = np.random.uniform(0, 1, n_samples)
# df['usage_spike'] = np.random.uniform(0, 1, n_samples)
# df['usage_drop_ratio'] = np.random.uniform(0, 1, n_samples)

# df['voltage'] = np.random.normal(220, 10, n_samples)
# df['current'] = np.random.normal(10, 3, n_samples)
# df['power_factor'] = np.random.uniform(0.7, 1.0, n_samples)
# df['frequency'] = np.random.normal(50, 0.5, n_samples)
# df['voltage_fluctuation'] = np.random.uniform(0, 1, n_samples)

# df['billing_amount'] = df['avg_consumption'] * np.random.uniform(0.1, 0.3, n_samples)
# df['avg_bill_last_3_months'] = df['billing_amount'] + np.random.normal(0, 50, n_samples)
# df['bill_difference'] = df['billing_amount'] - df['avg_bill_last_3_months']
# df['payment_delay_days'] = np.random.randint(0, 30, n_samples)
# df['unpaid_bills_count'] = np.random.randint(0, 5, n_samples)

# df['neighbor_avg_consumption'] = df['avg_consumption'] + np.random.normal(0, 40, n_samples)
# df['deviation_from_area'] = df['avg_consumption'] - df['neighbor_avg_consumption']
# df['line_loss'] = np.random.uniform(0, 0.3, n_samples)
# df['transformer_load'] = np.random.uniform(0.5, 1.5, n_samples)

# df['tamper_flag'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# # TARGET
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
# df.to_csv(r"F:\Python course\machineLearning\finalProject\electricity_theft_data_p3.csv", index=False)

# 2. LOAD FROM CSV
# ============================================================
df = pd.read_csv(r"F:\Python course\machineLearning\finalProject\electricity_theft_data_p3.csv")

# 3. FEATURE ENGINEERING
# ============================================================

df['power'] = df['voltage'] * df['current']
df['consumption_ratio'] = df['peak_consumption'] / df['avg_consumption']
df['consumption_change'] = (df['avg_consumption'] - df['previous_month_consumption']) / df['previous_month_consumption']

df['risk_score'] = df['usage_drop_ratio'] * 0.4 + df['tamper_flag'] * 0.6

df['bill_per_unit'] = df['billing_amount'] / (df['avg_consumption'] + 1)

# ============================================================
# 4. Z-SCORE CLEANING
# ============================================================

numeric_cols = df.drop('theft', axis=1).select_dtypes(include=[np.number]).columns
z_scores = np.abs(zscore(df[numeric_cols]))

df_clean = df[(z_scores < 3).all(axis=1)]

if df_clean['theft'].nunique() > 1:
    df = df_clean
else:
    print("Skipping Z-score cleaning")

# ============================================================
# 5. SIMPLE VISUAL (IMPORTANT)
# ============================================================

plt.figure()
plt.scatter(df['avg_consumption'], df['billing_amount'])
plt.title("Consumption vs Billing")
plt.xlabel("Consumption")
plt.ylabel("Billing")
plt.show()

# ============================================================
# 6. SPLIT
# ============================================================

X = df.drop('theft', axis=1)
y = df['theft']

# ============================================================
# 7. SMOTE
# ============================================================

# BEFORE
plt.figure()
y.value_counts().plot(kind='bar')
plt.title("Before SMOTE")
plt.show()

smote = SMOTE()
X, y = smote.fit_resample(X, y)

# AFTER
plt.figure()
pd.Series(y).value_counts().plot(kind='bar')
plt.title("After SMOTE")
plt.show()

# ============================================================
# 8. SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============================================================
# 10. MODEL
# ============================================================

model = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# ============================================================
# 11. EVALUATION
# ============================================================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# CONFUSION MATRIX (VISUAL)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

plt.figure()
plt.barh(df.drop('theft', axis=1).columns, model.feature_importances_)
plt.title("Feature Importance")
plt.show()

# ============================================================
# 13. PREDICTION
# ============================================================

sample = X_test[0].reshape(1, -1)
print("Prediction:", model.predict(sample)[0])