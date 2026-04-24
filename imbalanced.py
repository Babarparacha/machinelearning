"""An imbalanced dataset is one where the classes (categories) are not represented equally.
 In other words, some classes have many more samples than others. This is common in
   classification problems.
   Why It Matters
If you train a model on this dataset without addressing imbalance, the model may:
Predict all transactions as legitimate
Achieve high accuracy (95%) but fail to detect fraud
This is misleading because accuracy alone doesn’t show performance on the minority class."""

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create a simulated imbalanced dataset
# 9500 legitimate transactions, 500 fraudulent
n_legit = 9500
n_fraud = 500

# Features: amount and random feature
np.random.seed(42)
amount_legit = np.random.normal(50, 10, n_legit)
amount_fraud = np.random.normal(200, 50, n_fraud)

feature1_legit = np.random.normal(0, 1, n_legit)
feature1_fraud = np.random.normal(1, 1, n_fraud)

# Combine into dataset
X = np.concatenate([
    np.column_stack((amount_legit, feature1_legit)),
    np.column_stack((amount_fraud, feature1_fraud))
])

y = np.array([0]*n_legit + [1]*n_fraud)  # 0=Legit, 1=Fraud

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))