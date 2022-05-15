from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics.loss_functions import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        n_samples, n_features = X.shape
        if self.include_intercept_:
            # Pad the matrix with a left column of ones
            X = np.pad(X, pad_width=((0, 0), (1, 0)), mode='constant', constant_values=1)
            assert X.shape == (n_samples, n_features + 1)
        self.coefs_ = pinv(X) @ y
        if self.include_intercept_:
            assert self.coefs_.shape == (n_features + 1,)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        n_samples, n_features = X.shape
        w = self.coefs_[1:] if self.include_intercept_ else self.coefs_
        assert w.shape == (n_features,)
        return X @ w

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self._predict(X))
