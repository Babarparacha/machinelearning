import pandas as pd
from sklearn.preprocessing import OneHotEncoder
"""when you get a categorial data and want to convert it to numerical data , the
process is called encoding"""
"""One-hot encoding is a data preprocessing technique in machine learning used to convert
 nominal (unordered) categorical data into a numerical format
   that algorithms can understand and process. """

df=pd.read_csv(r"F:\Python course\machineLearning\employees_demo_data.csv")
# print(df.head())
# print(df.isnull().sum())
#======fill gender column empty values=============
df['gender'].fillna(df['gender'].mode()[0],inplace=True)
print(df.head())


#=========**encode column gender** two methods====
#======= 1- get dummied method ========
# en_data=df['gender']
# print(en_data)
# en=pd.get_dummies(en_data)
# print(en) #=====it's return true or false values

#=======2- using scikit-learn method
ohe=OneHotEncoder()
en_data=df[['gender']]
ar=ohe.fit_transform(en_data)

# =====Convert to DataFrame for readability=======
encoded_df=pd.DataFrame(ar,columns=['gender'])
print(encoded_df)