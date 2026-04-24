"""Feature scaling (standardization), or Z-score normalization, is a preprocessing 
technique that transforms numerical data to have a mean of 0 and a standard deviation
 of 1, effectively scaling features to a common range. It is crucial for algorithms
   sensitive to data scale, like PCA, SVM, and KNN, ensuring no single feature dominates
     the model due to its magnitude."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r"F:\Python course\machineLearning\loan_dataset.csv")
# print(df.head())
# print(df.isnull().sum()) #==check null values
#========fill missing or null values =========
df['applicationIncome'].fillna(df['applicationIncome'].mean(),inplace=True)
#========plot sns graph ro check data nature
# sns.displot(df['applicationIncome'])
# plt.show()
# print(df.describe())
#============ initilize StandardScaler
ss=StandardScaler()
ss.fit(df[['applicationIncome']]) #======= train the model

# data=ss.transform(df[['applicationIncome']]) #========after apply result
# print(data)
#===========add changes in data set=====
df['applicationIncome_ss']=pd.DataFrame(ss.transform(df[['applicationIncome']]),columns=['x'])
# print(df.describe())
# print(df.head())
plt.figure(figsize=(12,5))
plt.title('before')
sns.displot(df['applicationIncome'])
plt.subplot(1,2,2)
plt.title('after')
sns.displot(df['applicationIncome_ss'])
plt.show()