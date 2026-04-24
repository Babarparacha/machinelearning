import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df=pd.read_csv(r"F:\Python course\machineLearning\employees_demo_data.csv")
#==========fil data with 10 value ==========
# df.fillna(10,inplace=True) 
# print(df.head())
# # df.info() #====type data======== 
# #=========backward filling==neche wala amount uper a jaye ga =======
# df.fillna(method="bfill",axis=1,inplace=True) 
# print(df.head())
# #=========forward filling==uper wala amount beche a jaye ga =======
# df.fillna(method="ffill",axis=1,inplace=True)
# print(df.head())
# #=======fill data using mode in a column=====
# df['rating'].fillna(df['rating'].mode()[0],inplace=True)
# print(df.head())
# ==============Select only the columns in the DataFrame that have data type 'object'
# ==========='object' usually means text/string data in pandas
for i in df.select_dtypes(include="object").columns:
 df[i].fillna(df[i].mode()[0],inplace=True)
print(df.head())
