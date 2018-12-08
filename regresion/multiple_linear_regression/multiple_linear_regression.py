from data_preprocessing import data_preprocessing
from sklearn.linear_model import LinearRegression


def _main():
    features, labels = data_preprocessing.import_dataset('50_Startups.csv', slice(0, 4), 4)

    features, _ = data_preprocessing.one_hot_encode_categorical_features(features, [3])

    features_train, features_test, labels_train, labels_test = \
        data_preprocessing.split_train_test(features, labels, test_size=0.2)

    regressor = LinearRegression()
    regressor.fit(features_train, labels_train)

    labels_test_pred = regressor.predict(features_test)


if __name__ == '__main__':
    _main()
