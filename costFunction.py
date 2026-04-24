"""What is a Cost Function?

A cost function (also called loss function) measures how wrong your model’s predictions are 
compared to the actual values.
Why do we need it?
When your model predicts values, they won’t be perfect.
The cost function gives a single number that represents total error.
🔵 Small cost → Good model
🔴 Large cost → Bad model
types of cost function(regression cost function)
1-MSE(mean square error) also known as L2 
2-RMSE(root mean square error)
3-MAE(mean absolute error)
4-R^2(accuracy) 
type of cost function in claaaification:
1- binary classification(0-1)
2-multi-class classification(allocated to one of more than classes )
Most Common Cost Function (Linear Regression)
We use Mean Squared Error (MSE):

j=n1​∑i=1n​(yi​−y^​i​)2
Meaning of formula:
yi=actual value
y^i=predcited value
n = number of data points
Square → makes all errors positive + punishes big errors more
How Cost Function Works (Step-by-Step)
Model makes predictions
Cost function calculates error
Algorithm checks: “Error kitna hai?”
Model adjusts itself to reduce error
Repeat until error becomes minimum

👉 This process is called optimization"""