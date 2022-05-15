from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, indices, counts = np.unique(y, return_inverse=True, return_counts=True)
        (n_samples, n_features), n_classes = X.shape, len(self.classes_)

        # self.mu_ = np.zeros((n_classes, n_features))
        # np.add.at(self.mu_, indices, X)
        # self.mu_ /= counts[:, None]
        # assert self.mu_.shape == (n_classes, n_features)

        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        for i, k in enumerate(self.classes_):
            self.mu_[i] = np.mean(X[y == k], axis=0)
            # self.vars_[i] = np.var(X[y == k], axis=0)   # biased
            self.vars_[i] = np.var(X[y == k], axis=0, ddof=1)   # unbiased
        assert self.mu_.shape == self.vars_.shape == (n_classes, n_features)

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

        lls = np.zeros((n_samples, n_classes))
        for k in range(n_classes):
            lls[:, k] = np.log(self.pi_[k])
            lls[:, k] += -0.5 * np.sum(np.log(2.0 * np.pi * self.vars_[k]))
            lls[:, k] -= 0.5 * np.sum(((X - self.mu_[k]) ** 2) / (self.vars_[k]), axis=1)

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
