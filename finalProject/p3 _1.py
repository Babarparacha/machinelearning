# ============================================================
# ELECTRICITY THEFT DETECTION - COMPLETE ML PROJECT (CSV BASED)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 1. DATA COLLECTION (CREATE CSV FILE)
# ============================================================

np.random.seed(42)
n_samples = 2000

df = pd.DataFrame()

# -------------------------------
# Core consumption features
# -------------------------------
df['avg_consumption'] = np.random.normal(300, 100, n_samples).clip(50)
df['peak_consumption'] = df['avg_consumption'] + np.random.normal(60, 30, n_samples)
df['previous_month_consumption'] = df['avg_consumption'] + np.random.normal(0, 50, n_samples)

df['sudden_drop'] = np.random.uniform(0, 1, n_samples)
df['night_day_ratio'] = np.random.uniform(0.5, 2.5, n_samples)

# -------------------------------
# Electrical features
# -------------------------------
df['voltage'] = np.random.normal(220, 10, n_samples)
df['current'] = np.random.normal(10, 3, n_samples)

# -------------------------------
# Billing & behavior
# -------------------------------
df['billing_amount'] = df['avg_consumption'] * np.random.uniform(0.1, 0.3, n_samples)
df['payment_delay_days'] = np.random.randint(0, 30, n_samples)

df['meter_variance'] = np.random.uniform(0, 1, n_samples)

# -------------------------------
# Grid / technical features
# -------------------------------
df['load_factor'] = df['avg_consumption'] / df['peak_consumption']
df['phase_imbalance'] = np.random.uniform(0, 1, n_samples)
df['transformer_load'] = np.random.uniform(0.5, 1.5, n_samples)
df['line_loss'] = np.random.uniform(0, 0.3, n_samples)

# -------------------------------
# Environmental
# -------------------------------
df['temperature'] = np.random.uniform(20, 45, n_samples)
df['humidity'] = np.random.uniform(30, 80, n_samples)

# -------------------------------
# Categorical encoded
# -------------------------------
df['area_type'] = np.random.choice([0, 1], n_samples)  # 0 rural, 1 urban
df['connection_type'] = np.random.choice([0, 1], n_samples)  # 0 residential, 1 commercial
df['tamper_flag'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# ============================================================
# TARGET CREATION (THEFT LOGIC)
# ============================================================
theft_score = (
    0.3 * df['sudden_drop'] +
    0.2 * df['tamper_flag'] +
    0.2 * df['meter_variance'] +
    0.2 * df['line_loss'] +
    0.1 * (1 - df['load_factor'])
)

df['theft'] = (theft_score > 0.6).astype(int)

# ============================================================
# SAVE TO CSV
# ============================================================
csv_path = r"F:\Python course\machineLearning\finalProject\electricity_theft_data_1.csv"
df.to_csv(csv_path, index=False)

print("✅ CSV file created successfully!")

# ============================================================
# 2. READ DATA FROM CSV
# ============================================================
data = pd.read_csv(csv_path)

print("\nFirst 5 rows:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

data['consumption_ratio'] = data['peak_consumption'] / data['avg_consumption']
data['power'] = data['voltage'] * data['current']

data['consumption_change'] = (
    data['avg_consumption'] - data['previous_month_consumption']
) / data['previous_month_consumption']

data['bill_per_unit'] = data['billing_amount'] / data['avg_consumption']

data['risk_score'] = (
    data['sudden_drop'] * 0.5 +
    data['tamper_flag'] * 0.5
)

data['env_impact'] = data['temperature'] * data['humidity']

print("\n✅ Feature Engineering Completed")

# ============================================================
# 4. SPLIT DATA
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
# 7. MODEL TRAINING
# ============================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Training Completed")

# ============================================================
# 8. EVALUATION
# ============================================================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ============================================================
# 9. VISUALIZATION
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
# 10. PREDICTION (NEW DATA)
# ============================================================

sample = X.iloc[[0]]
sample_scaled = scaler.transform(sample)

print("\n🔍 Prediction (0=Normal, 1=Theft):", model.predict(sample_scaled)[0])