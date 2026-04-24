# ============================================================
# ELECTRICITY THEFT DETECTION - COMPLETE ML PROJECT (FINAL)
# ============================================================

# ================================
# IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================
# 1. DATA COLLECTION (CREATE CSV)
# ================================
# Generate synthetic dataset (1500 samples)

# np.random.seed(42)
# n_samples = 1500

# customer_id = np.arange(1, n_samples + 1)

# avg_consumption = np.random.normal(300, 120, n_samples)
# avg_consumption = np.clip(avg_consumption, 50, None)

# peak_consumption = avg_consumption + np.random.normal(60, 25, n_samples)

# sudden_drop = np.random.uniform(0, 1, n_samples)
# night_day_ratio = np.random.uniform(0.5, 2.5, n_samples)

# voltage = np.random.normal(220, 10, n_samples)
# current = np.random.normal(10, 3, n_samples)

# tamper_flag = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])

# # Theft logic (simulated)
# prob_theft = (0.6 * sudden_drop) + (0.4 * tamper_flag)
# theft = (prob_theft > 0.65).astype(int)

# Create DataFrame
# df = pd.DataFrame({
#     'customer_id': customer_id,
#     'avg_consumption': avg_consumption,
#     'peak_consumption': peak_consumption,
#     'sudden_drop': sudden_drop,
#     'night_day_ratio': night_day_ratio,
#     'voltage': voltage,
#     'current': current,
#     'tamper_flag': tamper_flag,
#     'theft': theft
# })

# Save CSV
# df.to_csv('electricity_theft_data.csv', index=False)
# print("✅ CSV file created successfully!")
# exit()
# ================================
# 2. DATA PREPROCESSING
# ================================
data = pd.read_csv(r'F:\Python course\machineLearning\finalProject\electricity_theft_data.csv')

# Check data
print("\nFirst 5 rows:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())

# Drop unnecessary column
data = data.drop('customer_id', axis=1)

# ================================
# 3. FEATURE ENGINEERING
# ================================

# Create new features
data['consumption_ratio'] = data['peak_consumption'] / data['avg_consumption']
data['power'] = data['voltage'] * data['current']
data['risk_score'] = (data['sudden_drop'] * 0.7) + (data['tamper_flag'] * 0.3)
data['interaction'] = data['night_day_ratio'] * data['sudden_drop']

print("\n✅ Feature Engineering Completed")

# Split features and target
X = data.drop('theft', axis=1)
y = data['theft']

# ================================
# 4. FEATURE SCALING
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 5. TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================
# 6. MODEL TRAINING
# ================================
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
model.fit(X_train, y_train)

print("\n✅ Model Training Completed")

# ================================
# 7. MODEL EVALUATION
# ================================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================================
# 8. VISUALIZATION
# ================================

# Feature Importance
plt.figure()
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Theft Distribution
plt.figure()
data['theft'].value_counts().plot(kind='bar')
plt.title("Theft vs Non-Theft")
plt.xlabel("Class (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# Sudden Drop vs Theft
plt.figure()
plt.scatter(data['sudden_drop'], data['theft'])
plt.title("Sudden Drop vs Theft")
plt.xlabel("Sudden Drop")
plt.ylabel("Theft")
plt.show()

# Risk Score Distribution
plt.figure()
plt.hist(data['risk_score'], bins=30)
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.show()

# ================================
# 9. PREDICTION ON NEW DATA (FIXED)
# ================================

# Create sample input
sample = pd.DataFrame({
    'avg_consumption': [300],
    'peak_consumption': [380],
    'sudden_drop': [0.8],
    'night_day_ratio': [1.5],
    'voltage': [220],
    'current': [12],
    'tamper_flag': [1]
})

# Apply SAME feature engineering
sample['consumption_ratio'] = sample['peak_consumption'] / sample['avg_consumption']
sample['power'] = sample['voltage'] * sample['current']
sample['risk_score'] = (sample['sudden_drop'] * 0.7) + (sample['tamper_flag'] * 0.3)
sample['interaction'] = sample['night_day_ratio'] * sample['sudden_drop']

# Scale sample
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print("\n🔍 Prediction Result (1=Theft, 0=Normal):", prediction[0])

# ================================
# END OF PROJECT
# ================================