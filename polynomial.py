"""Polynomial regression is a type of regression where the relationship between independent
 variable (X) and dependent variable (Y) is modeled as an nth-degree polynomial, 
 instead of a straight line.
 formula 
y=b0​+b1​x+b2​x2+b3​x3+⋯+bn​xn  ============= here x has a power of x  but in multi linear it is not a power
 why we use it
 Why Use Polynomial Regression?
When data is not linear (curved pattern)
When linear regression gives poor accuracy
 """

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ================== Load Data ==================
# Read CSV file (make sure path is correct)
df = pd.read_csv(r"F:\Python course\machineLearning\polynomial.csv")

# ================== Check Data ==================
# print(df.head())  # View first 5 rows

# ================== Select Features ==================
# X should be 2D (important for sklearn)
x = df[['level']]   # independent variable (input)
y = df['salary']    # dependent variable (output)

# ================== Polynomial Conversion ==================
# Create polynomial features (degree=2 means x² included)
pf = PolynomialFeatures(degree=2)

# Fit and transform x into polynomial form
x_poly = pf.fit_transform(x)

# ================== Train Test Split ==================
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_poly, y, test_size=0.2, random_state=42
)

# ================== Train Model ==================
# Create Linear Regression model
lr = LinearRegression()

# Train model using training data
lr.fit(x_train, y_train)

# ================== Model Evaluation ==================
score = lr.score(x_test, y_test)
print("Model Accuracy (R²):", score)

# ================== Prediction ==================
# Predict values using full dataset
prd = lr.predict(x_poly)

# ================== Plot ==================
# Plot original data (scatter)
plt.scatter(df['level'], df['salary'], label='Original Data')

# Plot predicted curve (line)
plt.plot(df['level'], prd, color="red", label='Predicted Curve')

# Labels
plt.xlabel('Level')
plt.ylabel('Salary')

# Show legend properly
plt.legend()

# Show graph
plt.show()