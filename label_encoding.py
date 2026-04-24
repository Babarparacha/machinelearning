"""Label Encoding is a technique used in machine learning to convert categorical 
(text) data into numerical form by assigning each unique category a unique
 integer value."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
# df=pd.DataFrame({
#     "name":['haider','jameel','zahid','shahid']
# })
df=pd.read_csv(r"F:\Python course\machineLearning\employees_demo_data.csv")

le=LabelEncoder()
df['en_name']=le.fit_transform(df['name']) # this encode name column and give it a separate name
print(df)
