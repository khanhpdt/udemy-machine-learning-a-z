import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import data_preprocessing


def main():
    features, labels = data_preprocessing.import_dataset('Position_Salaries.csv', [1], [2])

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(features, labels)

    plt.scatter(features, labels, color='red', label='Training examples')

    # plot with high resolution because with only one feature, the simple plot will
    # show that all training examples are matched by the prediction, which is because
    # of the way decision tree regression predicts a value from an average of a region.
    feature_grid = np.arange(min(features), max(features), step=0.01)
    feature_grid = feature_grid.reshape((len(feature_grid), 1))
    plt.plot(feature_grid, regressor.predict(feature_grid), color='blue', label='Predictions')

    plt.legend()
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    main()
