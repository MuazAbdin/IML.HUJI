from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) \
        -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    ids = np.arange(X.shape[0])

    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)
    train_score, validation_score = .0, .0
    for fold_ids in folds:
        train_msk = ~np.isin(ids, fold_ids)
        fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])

        train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
        validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))

    return train_score / cv, validation_score / cv

    # groups = np.remainder(np.arange(y.size), cv)
    # train_score, validation_score = 0.0, 0.0
    # for k in range(cv):
    #     train_X, train_y = X[groups != k], y[groups != k]
    #     validate_X, validate_y = X[groups == k], y[groups == k]
    #     est = deepcopy(estimator).fit(train_X, train_y)
    #     train_score += scoring(train_y, est.predict(train_X))
    #     validation_score += scoring(validate_y, est.predict(validate_X))
    #
    # return train_score / cv, validation_score / cv

    # X_folds, y_folds = np.array_split(X, cv), np.array_split(y, cv)
    # train_score, validation_score = 0.0, 0.0
    # for i in range(cv):
    #     train_X = np.concatenate(X_folds[:i] + X_folds[i+1:])
    #     train_y = np.concatenate(y_folds[:i] + y_folds[i+1:])
    #     validate_X , validate_y = X_folds[i], y_folds[i]
    #     est = deepcopy(estimator).fit(train_X, train_y)
    #     # train_score += (scoring(train_y, est.predict(train_X)) / cv)
    #     # validation_score += (scoring(validate_y, est.predict(validate_X)) / cv)
    #     train_score += scoring(train_y, est.predict(train_X))
    #     validation_score += scoring(validate_y, est.predict(validate_X))
    #
    # return train_score / cv, validation_score / cv
