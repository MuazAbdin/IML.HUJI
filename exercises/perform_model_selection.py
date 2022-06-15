from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso, Ridge

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps
    # for eps Gaussian noise and split into training- and testing portions
    min_val, max_val = -1.2, 2
    x = np.linspace(min_val, max_val, num=100)
    y = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    x = np.random.rand(n_samples) * (max_val - min_val) + min_val
    dataset_x = x
    noiseless_dataset_y = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    dataset_y = noiseless_dataset_y + np.random.randn(n_samples) * noise
    train_X, train_y, test_X, test_y = split_train_test(X=pd.DataFrame(dataset_x),
                                                        y=pd.DataFrame(dataset_y),
                                                        train_proportion=2 / 3)
    train_X = train_X.to_numpy().reshape(-1)
    train_y = train_y.to_numpy().reshape(-1)
    test_X = test_X.to_numpy().reshape(-1)
    test_y = test_y.to_numpy().reshape(-1)

    colors = np.array(['#2596be', '#e28743', '#79be25'])
    symbols = np.array(['star', 'circle', 'triangle-up'])

    fig = go.Figure(data=[go.Scatter(x=np.linspace(min_val, max_val, num=100), y=y, mode="lines",
                                     marker={"color": colors[0], "size": 25},
                                     name='true model'),
                          go.Scatter(x=dataset_x, y=noiseless_dataset_y, mode="markers",
                                     marker={"color": colors[0]}, showlegend=False),
                          go.Scatter(x=train_X, y=train_y, mode="markers",
                                     marker={"color": colors[1]}, name='train set'),
                          go.Scatter(x=test_X, y=test_y, mode="markers",
                                     marker={"color": colors[2]}, name='test set')],
                    layout=go.Layout(
                        title=rf"$\text{{(1) Dataset with }} {n_samples} \text{{ samples and }} "
                              rf"{noise} \text{{ noise}}$",
                        xaxis_title=r"$x$", yaxis_title=r"$f(x)$", width=1000))
    fig.write_image(f"./ex5.figs/plot.samples({n_samples}-{noise}).png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_deg = 10
    train_scores = np.zeros(max_deg + 1)
    validation_scores = np.zeros(max_deg + 1)
    for k in range(max_deg + 1):
        tr_score, val_score = cross_validate(PolynomialFitting(k=k), X=train_X, y=train_y,
                                             scoring=mean_square_error)
        train_scores[k] = tr_score
        validation_scores[k] = val_score

    # print(train_scores.tolist())
    # print(validation_scores.tolist())

    min_ind = np.argmin(np.array(validation_scores))
    selected_k = np.arange(max_deg + 1)[min_ind]
    selected_error = validation_scores[min_ind]

    fig = go.Figure(data=[
        go.Scatter(name='Train Error', x=np.arange(max_deg + 1), y=train_scores,
                   mode='markers+lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name='Validation Error', x=np.arange(max_deg + 1), y=validation_scores,
                   mode='markers+lines', marker_color='rgb(220,179,144)'),
        go.Scatter(name='Selected Model', x=[selected_k], y=[selected_error],
                   mode='markers', marker=dict(color='darkred', symbol="x", size=10))],
        layout=go.Layout(
            title=rf"$\text{{(2) PolyFit Errors - }}"
                  rf"  m={n_samples}, \sigma^{{2}}={noise}$",
            xaxis_title=r"$k\text{ - Polynomial Degree}$",
            yaxis_title=r"$\text{Error Value}$",
            width=1000))

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    test_errors = [mean_square_error(y_true=test_y, y_pred=PolynomialFitting(k).fit(
        train_X, train_y).predict(test_X)) for k in range(max_deg + 1)]
    test_error = test_errors[selected_k]
    min_test_k = np.argmin(test_errors)

    print(f'- Fitting a model on {n_samples} samples with noise = {noise}:\n'
          f'\tMinimizing degree (k) = {selected_k} with generalization error = {test_error:.2f}. '
          f'The validation error was {selected_error:.2f}\n'
          f'\tWhile actual minimal generalization error = {test_errors[min_test_k]:.2f} '
          f'at k = {min_test_k}. ')

    fig.add_traces(data=[
        go.Scatter(name="Test Error", x=np.arange(max_deg + 1), y=test_errors,
                   mode='markers + lines', marker_color='rgb(25,115,132)'),
        go.Scatter(name='Minimizer', x=[min_test_k], y=[test_errors[min_test_k]],
                   mode='markers', marker=dict(color='tomato', symbol="star", size=10))])

    fig.write_image(f"./ex5.figs/plot.errors({n_samples}-{noise}).png")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting
    regularization parameter values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples, :], y[:n_samples]
    test_X, test_y = X[n_samples:, :], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter
    # for Ridge and Lasso regressions
    lam_range = np.linspace(0.001, 2, num=n_evaluations)
    avg_train_score_ridge = np.zeros(n_evaluations)
    avg_validation_score_ridge = np.zeros(n_evaluations)
    avg_train_score_lasso = np.zeros(n_evaluations)
    avg_validation_score_lasso = np.zeros(n_evaluations)
    for k, lam in enumerate(lam_range):
        ridge_tr_score, ridge_val_score = cross_validate(RidgeRegression(lam=lam),
                                                         X=train_X, y=train_y,
                                                         scoring=mean_square_error)
        avg_train_score_ridge[k] = ridge_tr_score
        avg_validation_score_ridge[k] = ridge_val_score

        lasso_tr_score, lasso_val_score = cross_validate(Lasso(alpha=lam),
                                                         X=train_X, y=train_y,
                                                         scoring=mean_square_error)
        avg_train_score_lasso[k] = lasso_tr_score
        avg_validation_score_lasso[k] = lasso_val_score

    model_names = [r"$\text{Ridge Regressions}$", r"$\text{Lasso Regressions}$"]
    fig = make_subplots(rows=2, cols=1, subplot_titles=model_names, vertical_spacing=0.1)
    fig.update_layout(title=rf"$\text{{(3) Ridge & Lasso Errors}}$",
                      xaxis2_title=r"$\lambda$",
                      yaxis1_title=r'$\text{Error Value}$', yaxis2_title=r'$\text{Error Value}$',
                      legend_tracegroupgap=300, width=800, height=700)
    fig.add_traces(data=[
        go.Scatter(name='Train Error', x=lam_range, y=avg_train_score_ridge,
                   mode='lines', marker_color='rgb(72,158,74)', legendgroup='1'),
        go.Scatter(name='Validation Error', x=lam_range, y=avg_validation_score_ridge,
                   mode='lines', marker_color='rgb(54,136,216)', legendgroup='1'),
        go.Scatter(name='Train Error', x=lam_range, y=avg_train_score_lasso,
                   mode='lines', marker_color='rgb(210,61,32)', legendgroup='2'),
        go.Scatter(name='Validation Error', x=lam_range, y=avg_validation_score_lasso,
                   mode='lines', marker_color='rgb(236,215,38)', legendgroup='2')],
        rows=[1, 1, 2, 2], cols=[1, 1, 1, 1])

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambda_ridge = lam_range[np.argmin(avg_validation_score_ridge)]
    best_lambda_lasso = lam_range[np.argmin(avg_validation_score_lasso)]
    print(f'The regularization parameter value that achieved the best validation errors '
          f'for:\n\t(1) The Ridge Regressions = {best_lambda_ridge:.3f} with error = '
          f'{np.min(avg_validation_score_ridge):.3f}\n\t(2) The Lasso Regressions = '
          f'{best_lambda_lasso:.3f} with error = {np.min(avg_validation_score_lasso):.3f}')

    fig.add_vline(x=best_lambda_ridge, line_width=2, line_dash="dash", line_color="#873e23",
                  annotation_text=rf"$\widehat{{\lambda}}^{{Ridge}}={best_lambda_ridge:.3f}$",
                  annotation_position="top right",
                  annotation_font_size=20,
                  annotation_font_color="#873e23",
                  row=1, col=1)
    fig.add_vline(x=best_lambda_lasso, line_width=2, line_dash="dash", line_color="#873e23",
                  annotation_text=rf"$\widehat{{\lambda}}^{{Lasso}}={best_lambda_lasso:.3f}$",
                  annotation_position="top right",
                  annotation_font_size=20,
                  annotation_font_color="#873e23",
                  row=2, col=1)
    fig.update_yaxes(type="log")
    fig.write_image(f"./ex5.figs/plot.ridge.lasso.errors.png")

    ridge_est = RidgeRegression(lam=best_lambda_ridge).fit(train_X, train_y)
    ridge_general_error = ridge_est.loss(test_X, test_y)
    lasso_est = Lasso(alpha=best_lambda_lasso).fit(train_X, train_y)
    lasso_general_error = mean_square_error(test_y, lasso_est.predict(test_X))
    ols_est = LinearRegression().fit(train_X, train_y)
    ols_general_error = ols_est.loss(test_X, test_y)

    print(f"Test Errors of the fitted models:\n"
          f"\t(1) Ridge Regression = {ridge_general_error:.3f}\n"
          f"\t(2) Lasso Regression = {lasso_general_error:.3f}\n"
          f"\t(3) OLS Regression = {ols_general_error:.3f}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    # select_polynomial_degree(n_samples=1500, noise=5)

    select_regularization_parameter()
