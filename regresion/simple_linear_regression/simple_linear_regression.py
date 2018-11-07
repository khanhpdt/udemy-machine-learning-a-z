from data_preprocessing import data_preprocessing
from sklearn.linear_model import LinearRegression


def _main():
    X, y = data_preprocessing.import_dataset('Salary_Data.csv', slice(0, -1), 1)

    X_train, X_test, y_train, y_test = data_preprocessing.split_train_test(X, y, 1 / 3)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)


if __name__ == '__main__':
    _main()
