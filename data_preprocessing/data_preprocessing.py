import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


def import_dataset(dataset_file, feature_column_idxs, label_column_idx=None):
    dataset = pd.read_csv(dataset_file)
    features = dataset.iloc[:, feature_column_idxs].values
    
    if label_column_idx is not None:
        labels = dataset.iloc[:, label_column_idx].values
        return features, labels
    
    return features


def feature_scaling(training_data, test_data):
    # here we scale the dummy variables created by the dummy encoding too.
    # but in some cases, we might not need to do that because their scale
    # is small (they can only be 0 or 1) and the benefit is that we can
    # preserve the correlations b/w those variables.
    training_data = standard_scale(training_data)
    test_data = standard_scale(test_data)

    # B/c the output is categorical data, we don't need to scale it.

    return training_data, test_data


def standard_scale(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def split_train_test(features, labels, test_size):
    features_for_training, features_for_test, labels_for_training, labels_for_test = \
        train_test_split(features, labels, test_size=test_size, random_state=0)
    return features_for_training, features_for_test, labels_for_training, labels_for_test


def handle_missing_data(data, columns):
    # replace missing data by the mean of the feature
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

    # only concern the columns with missing data
    imputer = imputer.fit(data[:, columns])

    data[:, columns] = imputer.transform(data[:, columns])


def one_hot_encode_categorical_features(features,
                                        categorical_feature_column_idxs,
                                        avoid_dummy_encoding_trap=True,
                                        labels=None):

    # If we just simply encode the categorical data to numbers,
    # this introduces the order relationship between the data. e.g., here we
    # encode the countries into numbers like 0, 1, 2, ..., and this has a
    # side effect that it introduces ordering into the countries which does
    # not make sense. This can be fixed by using the OneHotEncoder as below:
    column_transformer = make_column_transformer(
        (categorical_feature_column_idxs, OneHotEncoder()), remainder="passthrough")
    features = column_transformer.fit_transform(features)

    # remove one of the dummy variables, because it is recommended not to include
    # all dummy variables due their interdependency.
    if avoid_dummy_encoding_trap:
        features = features[:, 1:]

    # It's ok to just encode the categorical data in the output into numbers
    # and not necessary to use OneHotEncoder.
    # This is because ordering in the label will not affect a learning model.
    if labels is not None:
        label_encoder_for_label = LabelEncoder()
        labels = label_encoder_for_label.fit_transform(labels)

    return features, labels
