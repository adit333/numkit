"""Includes implementation of logistic regression algorithm.

Implementation of softmax regression using batch gradient descent.

Reference: Hands-On Machine Learning, Aurélien Géron
"""


import numpy as np


def softmax(logits):
    # The dimensions and interpretation of the logits matrix is above
    # It is an m * k matrix, with each row containing the scores for instance i
    exps = np.exp(logits)
    exp_sums = exps.sum(axis=1, keepdims=True)
    return exps / exp_sums

def to_one_hot(y):
    return np.diag(np.ones(y.max() + 1))[y]


class LogisticRegression:
    X: np.ndarray
    y: np.ndarray

    def __init__(self, eta: float = 0.5, n_epochs: int = 5000, eps: float = 1e-5) -> None:
        self.eta = eta
        self.n_epochs = n_epochs
        self.eps = eps
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.m = len(self.X)
        n_inputs = self.X.shape[1]             # 3 (2 features + 1 bias)
        n_outputs = len(np.unique(self.y))     # 3 (there are 3 iris classes
        self.Theta = np.random.randn(n_inputs, n_outputs)
        Y_train_one_hot = to_one_hot(self.y)

        for epoch in range(self.n_epochs):
            logits  = self.X @ self.Theta
            Y_proba = softmax(logits)                   # This is an m * k matrix with each column containing probability of class j

            errors = Y_proba - Y_train_one_hot
            gradients = 1 / self.m * (self.X.T @ errors)    # Compute gradient

            self.Theta = self.Theta - self.eta * gradients             # Gradient descent step

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.Theta
        Y_proba = softmax(logits)
        return np.argmax(Y_proba, axis=1)
