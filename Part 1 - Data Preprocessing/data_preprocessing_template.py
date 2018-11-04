# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data:

from sklearn.preprocessing import Imputer
# replace missing data by the mean of the feature
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# only concern the columns with missing data
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode categorical data:

# encode the country feature
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
# However, if we just simply encode the categorical data to numbers,
# this introduces the order relationship between the data. e.g., here we
# encode the countries into numbers like 0, 1, 2, ..., and this has a
# side effect that it introduces ordering into the countries which does
# not make sense. This can be fixed by using the OneHotEncoder as below:
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()

# It's ok to just encode the categorical data in the output into numbers
# and not necessary to use OneHotEncoder.
# The reason is because the machine learning model knows that there is no
# order in the ouput. (how do they know???)
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

