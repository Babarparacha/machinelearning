"""Ordinal Encoding is a technique used in machine learning to convert 
categorical data intom numerical values,
 where the categories have a meaningful order or ranking.
 What is Ordinal Data?
Ordinal data is categorical data with a clear order, but the difference between values 
is not necessarily equal."""

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

data = pd.DataFrame({
    'size': ['Small', 'Medium', 'Large', 'Medium', 'Small']
})

encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
encoded = encoder.fit_transform(data[['size']])

print(pd.DataFrame(encoded, columns=['size_encoded']))