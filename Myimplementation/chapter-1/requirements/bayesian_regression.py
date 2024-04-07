from .regression import Regression
import numpy as np


class BayesRegression(Regression):
    """Bayesian regression model.
    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def is_prior_defined(self):
        return self.w_mean is not None and self.w_precision is not None

    def getprior(self, ndim: int):
        if self.is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, x_train, y_train):
        # mean_prev, precision_prev = self.getprior(np.size(x_train, 1))
        mean_prev, precision_prev = self.getprior(x_train.shape[1])
        w_precision = precision_prev + self.beta * x_train.T @ x_train
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * x_train.T @ y_train,
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, x: np.ndarray, return_std: bool = False, sample_size: int = None):
        if sample_size is not None:
            w_sample = np.random.multivariate_normal(self.w_mean, size=sample_size)
            y_sample = x @ w_sample.T
            return y_sample
        y = x @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(x @ self.w_cov * x, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y
