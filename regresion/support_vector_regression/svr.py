import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from data_preprocessing import data_preprocessing


def main():
    features, labels = data_preprocessing.import_dataset('Position_Salaries.csv', [1], [2])

    feature_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features)

    label_scaler = StandardScaler()
    labels = label_scaler.fit_transform(labels)
    labels = labels.flatten()

    regressor = SVR(kernel='rbf', gamma='scale')
    regressor.fit(features, labels)

    plt.scatter(features, labels, color='red', label='Training examples')
    plt.plot(features, regressor.predict(features), color='blue', label='Predictions')
    plt.legend()
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    main()
