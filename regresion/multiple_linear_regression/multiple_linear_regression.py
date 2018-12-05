from data_preprocessing import data_preprocessing


def _main():
    features, labels = data_preprocessing.import_dataset('50_Startups.csv', slice(0, 4), 4)

    features, _ = data_preprocessing.one_hot_encode_categorical_features(features, [3])

    features_for_training, features_for_test, labels_for_training, labels_for_test = \
        data_preprocessing.split_train_test(features, labels, test_size=0.2)

    pass
    # - fit training data
    # - test performance on test data
    # - implement backward elimination


if __name__ == '__main__':
    _main()
