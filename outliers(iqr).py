"""In machine learning, an outlier is a data point that is significantly different from the rest
 of the data.
Simple definition
An outlier is an observation that lies far away from other observations in the dataset.
📊 Example:
If most house prices are between $100k–$300k and one house is $2 million, that $2M point is an outlier."""


# ================================
# 1. Import Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 2. Create Dataset
# ================================
data = [120, 130, 125, 140, 135, 128, 500, 600, 115, 118]

df = pd.DataFrame(data, columns=['price'])

print("Original Data:")
print(df)

# ================================
# 3. Visualize BEFORE Removing Outliers
# ================================
plt.figure()
sns.boxplot(x=df['price'])
plt.title("Before Removing Outliers")
plt.show()

# ================================
# 4. Calculate IQR
# ================================
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1

# Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# ================================
# 5. Detect Outliers
# ================================
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

print("\nOutliers Found:")
print(outliers)

# ================================
# 6. Remove Outliers
# ================================
df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

print("\nCleaned Data:")
print(df_clean)

# ================================
# 7. Visualize AFTER Removing Outliers
# ================================
plt.figure()
sns.boxplot(x=df_clean['price'])
plt.title("After Removing Outliers")
plt.show()