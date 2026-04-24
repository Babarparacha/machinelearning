import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"F:\Python course\machineLearning\employees_demo_data.csv")
# print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum())
#=========== show data that is missing==========
sns.heatmap(dataset.isnull(), cbar=False)
plt.show()
#=====this one delete entire column======
# dataset.drop(columns=["loanamount"],inplace=True)
# print(dataset.head())
#=====this one delete entire row======
# dataset.dropna(inplace=True)
# print(dataset.head())
