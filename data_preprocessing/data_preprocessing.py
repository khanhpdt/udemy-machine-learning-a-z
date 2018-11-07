import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


def import_dataset(dataset_file, feature_indices, output_indices):
    dataset = pd.read_csv(dataset_file)
    features = dataset.iloc[:, feature_indices].values
    output = dataset.iloc[:, output_indices].values
    return features, output


def feature_scaling(training_data, test_data):
    scaler = StandardScaler()

    # here we scale the dummy variables created by the dummy encoding too.
    # but in some cases, we might not need to do that because their scale
    # is small (they can only be 0 or 1) and the benefit is that we can
    # preserve the # correlations b/w those variables.
    training_data = scaler.fit_transform(training_data)
    test_data = scaler.transform(test_data)

    # B/c the output is categorical data, we don't need to scale it.

    return training_data, test_data


def split_train_test(features, output, test_size):
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test


def handle_missing_data(data, columns):
    # replace missing data by the mean of the feature
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # only concern the columns with missing data
    imputer = imputer.fit(data[:, columns])
    data[:, columns] = imputer.transform(data[:, columns])


# todo: customize this method if needed
def encode_categorical_data(X, y):
    # encode the first feature
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
