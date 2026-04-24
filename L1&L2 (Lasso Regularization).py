import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np 

df=pd.read_csv(r"F:\Python course\machineLearning\housing_data_50_rows.csv")
# print(df.head())
#====apply cooef to check if linear regression can apply ??=====
# plt.figure(figsize=(10,10))
# sns.heatmap(data=df.corr(),annot=True)
# plt.show()
#=========separate depenedent and indepenedent ======
x=df.iloc[:,:-1]
y=df["price"]
sc=StandardScaler()
sc.fit(x)
x=pd.DataFrame(sc.transform(x),columns=x.columns)
# print(x)
# ================= TRAIN TEST SPLIT =================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
lr=LinearRegression()
lr.fit(x_train,y_train)
sr=lr.score(x_test,y_test)
#=========check error ==========
# mean_sq=mean_squared_error(y_test,lr.predict(x_test))
# print(mean_sq)
# mean_ab_error=(y_test,lr.predict(x_test))
# print(mean_ab_error)
# print(np.sqrt(mean_sq)) #======root mean square error 
# print(sr)
# plt.figure(figsize=(15,5))
# plt.bar(x.columns,lr.coef_)
# plt.title("Linear Regression")
# plt.xlabel("columns")
# plt.ylabel("coef")
# plt.show()
#=========now apply lasso===========
la=Lasso(alpha=1.0)
la.fit(x_train,y_train)
scor=la.score(x_test,y_test)
# # print(scor)
# plt.figure(figsize=(15,5))
# plt.title("Lasso")
# plt.xlabel("columns")
# plt.ylabel("coef")
# plt.bar(x.columns,la.coef_)
# plt.show()
#=========now apply ridge===========
ri=Ridge(alpha=1.0)
ri.fit(x_train,y_train)
scor=ri.score(x_test,y_test)
# print(scor)
# plt.figure(figsize=(15,5))
# plt.title("Ridge")
# plt.xlabel("columns")
# plt.ylabel("coef")
# plt.bar(x.columns,ri.coef_)
# plt.show()

#============ compare all coef =======
d=pd.DataFrame({"col_name":x.columns,"LinearRegression":lr.coef_,"Lasso":la.coef_,"ridge":ri.coef_})
print(d)

