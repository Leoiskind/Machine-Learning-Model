import numpy as np

class LinearRegression:

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        #Puts a column of 1s on the left of the feature matrix (X) so we can add a bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        #Splits intercept and coefficient calculated by normal equation
        self.intercept = theta[0]
        self.coefficients = theta
        self.feature_coefficients = self.coefficients[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coefficients

    def get_coefficients(self):
        return {
            'intercept': self.intercept,
            'slope': self.feature_coefficients[0] if len(self.feature_coefficients) > 0 else None
        }

    def score(self, X, y):
        y_pred = self.predict(X)

        #Residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)

        #Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        #R^2 score
        return 1 - ss_res/ss_tot