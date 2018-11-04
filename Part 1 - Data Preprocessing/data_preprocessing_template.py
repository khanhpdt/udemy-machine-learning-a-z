# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# taking care of missing data
from sklearn.preprocessing import Imputer
# replace missing data by the mean of the feature
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# only concern the columns with missing data
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])