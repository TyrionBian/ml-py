
import numpy as np 
from src.linear.regressor import Regressor

class BayesianRegressor(Regressor):
    """
    Bayesian regression model
    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _fit(self, X, t):
        if self.w_mean is not None:
            mean_prev = self.w_mean
        else:
            mean_prev = np.zeros(np.size(X, 1))
        
        if self.w_precision is not None:
            precision_prev = self.w_precision
        else:
            precision_prev = self.alpha * np.eye(np.size(X, 1))
        
        w_precision = precision_prev + self.beta * X.T @ X
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_conv = np.linalg.inv(self.w_precision)

    def _predict(self, X, return_std=False, sample_size=None):
        if isinstance(sample_size, int):
            w_sample = np.random.multivariate_normal(self.w_mean, self.w_conv, sample_size)
            y = X @ w_sample.T
            return y
        y = X @ self.w_mean
        if return_std:
            y_var = 1/self.beta + np.sum(X @ self.w_conv * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

