"""What is L1 (Lasso) Regularization in Machine Learning?

L1 regularization, also called Lasso (Least Absolute Shrinkage and Selection Operator), is a
 technique used to prevent overfitting in models (especially linear regression).

It does this by adding a penalty to the loss function based on the absolute values of the 
coefficients.
instead of minimizing just the error, Lasso minimizes:
Loss=MSE+λ∑∣wi​∣
MSE = Mean Squared Error
λ (lambda) = regularization strength
wᵢ = model coefficients
👉 The key idea: penalize large weights
"""
"""What is L2 (Ridge) Regularization?

L2 regularization, also called Ridge Regression, is a technique used to reduce overfitting by
 penalizing large model coefficients.

Instead of eliminating features (like L1), it shrinks coefficients smoothly toward zero.
Formula (Core Idea)

Ridge modifies the loss function by adding a penalty on squared weights:
Loss=MSE+λ∑i=1n​wi2^2
MSE = Mean Squared Error
λ (lambda / alpha) = regularization strength
wᵢ² = squared coefficients

👉 The bigger the weight → the bigger the penalty
"""