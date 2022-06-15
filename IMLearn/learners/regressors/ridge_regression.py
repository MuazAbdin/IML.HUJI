from __future__ import annotations
from typing import NoReturn, Tuple
from ...base import BaseEstimator
import numpy as np


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            (lambda): Regularization parameter to be used when fitting a model

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
        from . import LinearRegression
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam
        # self.lre_ = LinearRegression(include_intercept=include_intercept)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

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
            X = np.c_[np.ones(len(X)), X]
            assert X.shape == (n_samples, n_features + 1)
        u, s, vh = np.linalg.svd(X, full_matrices=True)
        s /= (s ** 2 + self.lam_)
        pad_m = np.max([u.shape[0] - s.shape[0], 0])
        pad_d = np.max([vh.shape[0] - s.shape[0], 0])
        smat = np.pad(np.diag(s), pad_width=((0, pad_m), (0, pad_d)),
                      mode='constant', constant_values=0)
        self.coefs_ = (u @ smat @ vh).T @ y
        if self.include_intercept_:
            assert self.coefs_.shape == (n_features + 1,)

        # # using: (X^T.X+λI)^−1 . X^T . y
        # I = np.identity(X.shape[1])
        # # adjusting the first value in I to be 0, to account for the intercept term
        # if self.include_intercept_:
        #     assert I.shape == (n_features + 1, n_features + 1)
        #     I[0][0] = 0
        # self.coefs_ = np.linalg.inv(X.T @ X + self.lam_ * I) @ X.T @ y

        # self.lre_.fit(*self.__transform(X, y))
        # self.coefs_ = self.lre_.coefs_

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
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]
        return X @ self.coefs_

        # return self.lre_.predict(X)

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
        from IMLearn.metrics.loss_functions import mean_square_error
        return mean_square_error(y_true=y, y_pred=self._predict(X))
        # return self.lre_.loss(X, y)

    # def __transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Transform X and y to be solved with OLS
    #     """
    #     n_samples, n_features = X.shape
    #     X = np.vstack((X, np.identity(n_features) * np.sqrt(self.lam_)))
    #     y = np.hstack((y, np.zeros(n_features)))
    #     assert X.shape == (n_samples + n_features, n_features) and \
    #            y.shape == (n_samples + n_features, )
    #     return X, y
