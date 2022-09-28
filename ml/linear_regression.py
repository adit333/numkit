"""Implementation of linear regression"""

import numpy as np


class LinearRegression:
    def __init__(self, optimizer: str = "gradient_descent", learning_rate: float = 0.01, iterations: int = 1000):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the data

        Arguments:
            X: The training data
            y: The target values
        """
        self.X = X
        self.y = y
        self._initialize_weights()
        self._optimize()

        if self.optimizer == "gradient_descent":
            self._gradient_descent()
        elif self.optimizer == "normal_equation":
            self._normal_equation()
    
    def _gradient_descent(self) -> None:
        """Perform gradient descent to optimize the weights"""
        for _ in range(self.iterations):
            self.weights -= self.learning_rate * self._gradient()
    
    def _gradient(self) -> np.ndarray:
        """Calculate the gradient"""
        y_pred = self.predict(self.X)
        return self.X.T.dot(y_pred - self.y) / self.X.shape[0]
    
    def _normal_equation(self) -> None:
        """Calculate the weights using the normal equation"""
        self.weights = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)

    def _initialize_weights(self) -> None:
        """Initialize the weights"""
        self.weights = np.zeros(self.X.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model

        Arguments:
            X: The data to predict

        Returns:
            The predictions as a numpy array
        """
        return X.dot(self.weights)
