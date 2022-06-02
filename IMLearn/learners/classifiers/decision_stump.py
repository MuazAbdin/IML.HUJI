from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is above the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = 0, 0, 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = X.shape[1]
        opt_err = np.inf
        for sign, j in product([-1, 1], range(n_features)):
            thr, thr_err = self._find_threshold(X[:, j], y, sign)
            if thr_err < opt_err:
                self.threshold_, self.j_, self.sign_ = thr, j, sign
                opt_err = thr_err
        self.fitted_ = True

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

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        n_samples, n_features = X.shape
        res = self.sign_ * ((X[:, self.j_] >= self.threshold_) * 2 - 1)
        assert res.shape == (n_samples,)
        return res

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) \
            -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign`
        whereas values which equal to or above the threshold are predicted as `sign`
        """
        sort_index = np.argsort(values)
        new_D = np.abs(labels)
        new_labels = np.sign(labels)
        sorted_values = values[sort_index]
        sorted_labels = new_labels[sort_index]
        new_D = new_D[sort_index]

        thrs = np.concatenate([[-np.inf], [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)], [np.inf]])
        min_loss = np.sum(new_D[sorted_labels == sign])
        losses = np.append(min_loss, min_loss - np.cumsum(new_D * (sorted_labels * sign)))
        min_ind = np.argmin(losses)
        thr_err = losses[min_ind]

        thr = np.inf
        thr = -np.inf if min_ind == 0 else thr
        thr = thrs[min_ind] if min_ind != values.shape[0] else thr

        return thr, thr_err
        # sort_idx = np.argsort(values)
        # values, labels = values[sort_idx], labels[sort_idx]
        # thrs = np.concatenate([[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        # min_loss = np.sum(labels == sign)
        # losses = np.append(min_loss, min_loss - np.cumsum(labels * sign))
        # min_loss_idx = np.argmin(losses)
        # return thrs[min_loss_idx], losses[min_loss_idx]

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
