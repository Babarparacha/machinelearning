"""Used when there are multiple input variables.
formula
y=b0​+b1​x1​+b2​x2​+...+bn​xn​
Example:
Predicting house price using:
Size
Number of bedrooms
"""

from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"F:\Python course\machineLearning\age_experience_salary.csv")
# print(df.head())
# print(df.isnull().sum())
#===========plot a diagram===========
sns. pairplot(data=df)
# plt.show()
#==========check linearty using correlation=========
sns.heatmap(data=df.corr(),annot=True)
# plt.show()
#========== predict on salary,age base==========
x=df.iloc[:,:-1]
# print(x)
#print(x.columns) #=====check x column
y=df['salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_test,y_test)*100
# print(f"Slope (m): {lr.coef_[0]}")
# print(f"Intercept (b): {lr.intercept_}")
#=========result on formula y_pred=mx+b=========
pre=lr.predict(x_test)
print(pre)