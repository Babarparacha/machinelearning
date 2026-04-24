# ================================
# 1. Import Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
plt.title("Before Removing Outliers (Z-Score)")
plt.show()

# ================================
# 4. Calculate Z-Score
# ================================
z_scores = np.abs(stats.zscore(df['price']))
print("\nZ-scores:\n", z_scores)

# Typical threshold: 3
threshold = 3
outliers = df[z_scores > threshold]

print("\nOutliers Detected (Z-score > 3):")
print(outliers)

# ================================
# 5. Remove Outliers
# ================================
df_clean = df[z_scores <= threshold]

print("\nCleaned Data:")
print(df_clean)

# ================================
# 6. Visualize AFTER Removing Outliers
# ================================
plt.figure()
sns.boxplot(x=df_clean['price'])
plt.title("After Removing Outliers (Z-Score)")
plt.show()