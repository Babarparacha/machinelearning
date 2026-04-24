"""it is a scaling technique in which values are shifted and rescaled so they end up ranging
between o and 1.it is also known as Min-Max scaling
xnew=xi-min(X)/max(x)-min(X)"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv(r"F:\Python course\machineLearning\loan_dataset.csv")
# print(df.head())
#========plot sns graph ro check data nature
# sns.displot(df['coApplication'])
# plt.show()
# print(df.describe())
#============ initilize StandardScaler
ms=MinMaxScaler()
ms.fit(df[['coApplication']]) #======= train the model
df['coApplication_min']=pd.DataFrame(ms.transform(df[['coApplication']]),columns=['x'])

data=ms.transform(df[['coApplication']]) #========after apply result
print(data)
print(df.head(3))
#==========create chart
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.title('before')
# sns.displot(df['coApplication'])
# plt.subplot(1,2,2)
# plt.title('after')
# sns.displot(df['coApplication_min'])
# plt.show()
# ==========2nd chart example=========
# # Plot before and after scaling
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title('Before Scaling')
sns.histplot(df['coApplication'], kde=True, bins=30)

plt.subplot(1,2,2)
plt.title('After Scaling')
sns.histplot(df['coApplication_min'], kde=True, bins=30)

plt.show() 