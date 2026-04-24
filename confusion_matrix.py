"""A Confusion Matrix is a fundamental tool for evaluating the performance of a classification
 model. It is essentially a $2 \times 2$ table (for binary classification) that compares the 
 actual target values with those predicted by the machine learning model.The Four QuadrantsTo 
 understand the metrics, you first need to identify the four outcomes of the matrix:True Positive 
 (TP): The model predicted "Positive" and it was actually "Positive.
 "True Negative (TN): The model predicted "Negative" and it was actually 
 "Negative."False Positive (FP): The model predicted "Positive" but it was actually 
 "Negative" (Type I Error).False Negative (FN): The model predicted "Negative" but it was
   actually "Positive" (Type II Error).
   1. Recall (Sensitivity)Recall (also known as Sensitivity) measures the model's ability
     to find all the positive instances. It answers the question: "Of all the people who actually
       have the disease, how many did we correctly identify?
       formula:Recall=Tp/Tp+FN
       When to prioritize: When the cost of a False Negative is high (e.g., missing a cancer 
       diagnosis).
2. Precision
Precision measures the accuracy of the positive predictions. It answers the question:
 "Of all the people the model predicted as having the disease, how many actually have it?"
 formula:precission=Tp/Tp+FP
 When to prioritize: When the cost of a False Positive is high (e.g., marking a legitimate email as "Spam"
 3. F1-Score
The F1-Score is the harmonic mean of Precision and Recall. It provides a single score that
 balances both metrics.
 formula:f1 score=2*precission*Recall/precission+Recall
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

df=pd.read_csv(r"F:\Python course\machineLearning\placement_data.csv")
# print(df.head())
x=df.iloc[:,:-1]  #====take all columns execpt last one
# print(x)
y=df['placed']  # our input
# ================= TRAIN TEST SPLIT =================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr=LogisticRegression()
lr.fit(x_train,y_train)
score=lr.score(x_test,y_test)
# print("="*50)
# print(score)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
cf = confusion_matrix(y_test, y_pred)
# Plot
sns.heatmap(cf, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

ps=precision_score(y_test,y_pred)*100
print(ps)
rs=recall_score(y_test,y_pred)*100
print(rs)
fs=f1_score(y_test,y_pred)*100
print(fs)