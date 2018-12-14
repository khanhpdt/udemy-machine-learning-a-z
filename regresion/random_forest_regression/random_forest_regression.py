import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import data_preprocessing


def main():
    features, labels = data_preprocessing.import_dataset('Position_Salaries.csv', [1], [2])
    labels = labels.flatten()

    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(features, labels)

    plt.scatter(features, labels, color='red', label='Training examples')

    feature_grid = np.arange(min(features), max(features), step=0.01)
    feature_grid = feature_grid.reshape((len(feature_grid), 1))
    plt.plot(feature_grid, regressor.predict(feature_grid), color='blue', label='Predictions')

    plt.legend()
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    main()
