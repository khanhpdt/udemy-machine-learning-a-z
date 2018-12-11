from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from data_preprocessing import data_preprocessing
import matplotlib.pyplot as plt


def main():
    features, labels = data_preprocessing.import_dataset('Position_Salaries.csv', [1], [2])

    polynomial_features = PolynomialFeatures(degree=4).fit_transform(features)

    regressor = LinearRegression()
    regressor.fit(polynomial_features, labels)

    plt.scatter(features, labels, color='red', label='Training examples')
    plt.plot(features, regressor.predict(polynomial_features), color='blue', label='Predictions')
    plt.legend()
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    main()
