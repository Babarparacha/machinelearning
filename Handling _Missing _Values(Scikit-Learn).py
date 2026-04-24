import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df=pd.read_csv(r"F:\Python course\machineLearning\employees_demo_data.csv")

#======= fill missing data using sklearn==========
# Numeric columns only
num_cols = ['rating', 'loanamount']
si = SimpleImputer(strategy="mean")
arr = si.fit_transform(df[num_cols])
#=========== Convert back to DataFrame==========
df_clean = pd.DataFrame(arr, columns=num_cols)
print(df_clean)
print(df_clean.isnull().sum())