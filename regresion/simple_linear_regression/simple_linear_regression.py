from data_preprocessing import data_preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def _main():
    X, y = data_preprocessing.import_dataset('Salary_Data.csv', slice(0, -1), 1)

    X_train, X_test, y_train, y_test = data_preprocessing.split_train_test(X, y, 1 / 3)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    visualize_performance_on_training(regressor, X_train, y_train)
    visualize_performance_on_test(regressor, X_test, y_test)


def visualize_performance_on_test(regressor, X_test, y_test):
    plt.scatter(X_test, y_test, color='red')
    # for this linear regression model, it is also correct to
    # plot using plt.plot(X_train, regressor.predict(X_train), color='blue')
    # because plt.plot(X_train, regressor.predict(X_train)) and
    # plt.plot(X_test, regressor.predict(X_test)) plot the same line
    # of the same regression model.
    plt.plot(X_test, regressor.predict(X_test), color='blue')
    plt.title('Salary vs. Experience (Test set)')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()


def visualize_performance_on_training(regressor, X_train, y_train):
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs. Experience (Training set)')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    _main()
