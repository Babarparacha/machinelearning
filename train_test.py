"""Train-test split is a model validation procedure in machine learning used to evaluate 
how well a model generalizes to new, unseen data. It involves dividing a single dataset into
 two distinct subsets: 

    Training Set: The larger portion of the data used to "fit" the model, allowing it to learn 
    patterns and relationships.
    Test Set: A separate portion held back from the model during training. It acts as a 
    "blind test" to assess the model's accuracy on data it has never encountered. 

Why is it important?
    The primary goal is to prevent overfitting, where a model simply memorizes the training data
    instead of learning general rules. Testing on the same data used for training would provide an 
    unfairly optimistic and misleading accuracy
  what is overfitting?
        Overfitting is a modeling error in machine learning that occurs when a model learns
        training data too well, capturing noise and specific patterns instead of general trends
 """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"F:\Python course\machineLearning\house_prediction_dataset.csv")
# print(df.head())
input_data=df.iloc[:,:-1]  #===selects all rows and all columns except the last one====
output_data=df['House_Price']
# print(input_data)
x_train,y_train,x_test,y_test=train_test_split(input_data,output_data,test_size=0.25)
# print(x_train)
# print("="*50)
# print(y_train)
# print("="*50)
print(x_test) #====split data from df['House_Price']=====
# print("="*50)
# print(y_test)   #====split data from df['House_Price']=====