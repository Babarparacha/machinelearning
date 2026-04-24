# =========================
# ELECTRICITY THEFT - EDA (FIXED VERSION)
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# =========================
# 1. LOAD DATA
# =========================
file_path = r'F:\Python course\machineLearning\finalProject\synthetic_meter_data.csv'

df = pd.read_csv(file_path)

# Convert date
df['date'] = pd.to_datetime(df['date'])

# Sort data (VERY IMPORTANT)
df = df.sort_values(by=["customer_id", "date"])

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nSample Data:\n", df.head())

# =========================
# 2. FEATURE ENGINEERING (🔥 FIX)
# =========================

print("\nCreating features...")

df["rolling_mean"] = df.groupby("customer_id")["units_consumed"].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

df["rolling_std"] = df.groupby("customer_id")["units_consumed"].transform(
    lambda x: x.rolling(window=7, min_periods=1).std()
)

df["diff"] = df["units_consumed"] - df["rolling_mean"]

df["usage_ratio"] = df["units_consumed"] / (df["rolling_mean"] + 1e-5)

df["day_of_week"] = df["date"].dt.dayofweek

# Handle NaN from std
df["rolling_std"].fillna(0, inplace=True)

print("✅ Features created successfully")

# =========================
# 3. BASIC INFO
# =========================
print("\nInfo:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# =========================
# 4. STATISTICS
# =========================
print("\nStatistical Summary:\n")
print(df.describe())

# =========================
# 5. CLASS DISTRIBUTION
# =========================
plt.figure()
df['is_theft'].value_counts().plot(kind='bar')
plt.title("Theft vs Normal Distribution")
plt.show()

# =========================
# 6. USAGE DISTRIBUTION
# =========================
plt.figure()
sns.histplot(df['units_consumed'], bins=50, kde=True)
plt.title("Units Consumed Distribution")
plt.show()

# =========================
# 7. THEFT VS NORMAL
# =========================
plt.figure()
sns.boxplot(x='is_theft', y='units_consumed', data=df)
plt.title("Usage Comparison")
plt.show()

# =========================
# 8. TIME SERIES
# =========================
sample_customer = df['customer_id'].sample(1).iloc[0]
subset = df[df['customer_id'] == sample_customer]

plt.figure()
plt.plot(subset['date'], subset['units_consumed'])
plt.title(f"Customer {sample_customer} Usage")
plt.xticks(rotation=45)
plt.show()

# =========================
# 9. ROLLING ANALYSIS
# =========================
plt.figure()
plt.plot(subset['date'], subset['units_consumed'], label='Actual')
plt.plot(subset['date'], subset['rolling_mean'], label='Rolling Mean')
plt.legend()
plt.title("Rolling Mean vs Actual")
plt.xticks(rotation=45)
plt.show()

# =========================
# 10. CORRELATION
# =========================
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()

# =========================
# 11. DAY PATTERN
# =========================
plt.figure()
sns.boxplot(x='day_of_week', y='units_consumed', data=df)
plt.title("Usage by Day")
plt.show()

# =========================
# 12. FRAUD SCORING (ENHANCED)
# =========================

def calculate_fraud_score(df):
    df = df.copy()

    features = ['units_consumed', 'rolling_mean', 'rolling_std', 'diff', 'usage_ratio']

    for col in features:
        df[col + '_z'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    df['risk_score'] = (
        0.35 * abs(df['units_consumed_z']) +
        0.25 * abs(df['diff_z']) +
        0.20 * abs(df['rolling_std_z']) +
        0.20 * abs(df['usage_ratio_z'])
    )

    # Normalize 0–100
    df['fraud_score'] = 100 * (df['risk_score'] - df['risk_score'].min()) / (
        df['risk_score'].max() - df['risk_score'].min() + 1e-6
    )

    return df

df = calculate_fraud_score(df)

# =========================
# 13. FRAUD ANALYSIS
# =========================
plt.figure()
sns.histplot(df['fraud_score'], bins=50, kde=True)
plt.title("Fraud Score Distribution")
plt.show()

# Top risky
print("\nTop 10 High Risk:")
print(df.sort_values(by='fraud_score', ascending=False).head(10)[
    ['customer_id', 'units_consumed', 'fraud_score']
])

# High risk filter
high_risk = df[df['fraud_score'] > 70]
print(f"\nHigh Risk Count (>70): {len(high_risk)}")

# Scatter
plt.figure()
plt.scatter(df['units_consumed'], df['fraud_score'], alpha=0.3)
plt.title("Usage vs Fraud Score")
plt.xlabel("Units")
plt.ylabel("Fraud Score")
plt.show()

# =========================
# FINAL SAVE (IMPORTANT)
# =========================
# df.to_csv("eda_processed_data.csv", index=False)

print("\n✅ EDA COMPLETED SUCCESSFULLY")