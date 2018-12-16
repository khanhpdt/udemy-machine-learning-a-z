from data_preprocessing import data_preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plots.classification_result_visualizer import visualize_two_feature_classification


features, labels = data_preprocessing.import_dataset(
        'datasets/Social_Network_Ads.csv', [2, 3], 4)

feature_scaler = StandardScaler()
features = feature_scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = \
    data_preprocessing.split_train_test(features, labels, test_size = 0.25)

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(features, labels)

visualize_two_feature_classification(features_train, labels_train, classifier, 
                                     xlabel='Age', ylabel='Estimated salary')
