import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_preprocessing():
    X, y = import_dataset()

    # disable these two steps for now
    # handle_missing_data(X)
    # X, y = encode_categorical_data(X, y)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train, X_test = feature_scaling(X_test, X_train)


def feature_scaling(X_test, X_train):
    sc_X = StandardScaler()
    # here we scale the dummy variables created by the dummy encoding too.
    # but in some cases, we might not need to do that because their scale
    # is small (they can only be 0 or 1) and the benefit is that we can
    # preserve the # correlations b/w those variables.
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # B/c the output is categorical data, we don't need to scale it.

    return X_train, X_test


def split_train_test(X, y):
    # Split the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def import_dataset():
    # Importing the dataset
    dataset = pd.read_csv('Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values
    return X, y


def handle_missing_data(X):
    # Taking care of missing data:
    # replace missing data by the mean of the feature
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # only concern the columns with missing data
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])


def encode_categorical_data(X, y):
    # Encode categorical data:
    # encode the country feature
    label_encoder_X = LabelEncoder()
    X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
    # However, if we just simply encode the categorical data to numbers,
    # this introduces the order relationship between the data. e.g., here we
    # encode the countries into numbers like 0, 1, 2, ..., and this has a
    # side effect that it introduces ordering into the countries which does
    # not make sense. This can be fixed by using the OneHotEncoder as below:
    one_hot_encoder = OneHotEncoder(categorical_features=[0])
    X = one_hot_encoder.fit_transform(X).toarray()
    # It's ok to just encode the categorical data in the output into numbers
    # and not necessary to use OneHotEncoder.
    # The reason is because the machine learning model knows that there is no
    # order in the ouput. (how do they know???)
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    return X, y


if __name__ == '__main__':
    run_preprocessing()
