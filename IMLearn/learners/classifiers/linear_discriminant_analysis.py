from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, indices, counts = np.unique(y, return_inverse=True, return_counts=True)
        (n_samples, n_features), n_classes = X.shape, len(self.classes_)

        self.mu_ = np.zeros((n_classes, n_features))
        np.add.at(self.mu_, indices, X)
        self.mu_ /= counts[:, None]
        assert self.mu_.shape == (n_classes, n_features)

        self.cov_ = np.zeros((n_features, n_features))
        for i, k in enumerate(self.classes_):
            self.cov_ += (X[y == k] - self.mu_[i]).T @ X[y == k] - self.mu_[i]
        self.cov_ /= (n_samples - n_classes)    # unbiased
        # self.cov_ /= n_samples  # biased

        # self.cov_ = np.cov(X.T, ddof=n_classes)
        self._cov_inv = inv(self.cov_)
        assert self.cov_.shape == self._cov_inv.shape == (n_features, n_features)

        self.pi_ = counts / n_samples
        assert self.pi_.shape == (n_classes,)

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
        res = self.classes_[np.argmax(self.likelihood(X), axis=1)]
        assert res.shape == (n_samples,)
        return res

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        (n_samples, n_features), n_classes = X.shape, len(self.classes_)

        c = - 0.5 * n_features * np.log(2 * np.pi) - 0.5 * np.log(det(self.cov_))
        beta = - 0.5 * np.diag(self.mu_ @ self._cov_inv @ self.mu_.T) + np.log(self.pi_)
        alpha = self._cov_inv @ self.mu_.T

        lls = X @ alpha + beta + c
        assert lls.shape == (n_samples, n_classes)
        return lls

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y_true=y, y_pred=self._predict(X))
