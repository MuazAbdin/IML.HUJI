import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from copy import deepcopy

from plotly.subplots import make_subplots

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.gradient_descent import OUTPUT_VECTOR_TYPE
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import plotly.io as pio

pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(margin=dict(l=20, r=20, t=40, b=0))
)
pio.templates.default = "simple_white+custom"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which
        regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=rf"$\text{{Gradient Descent Path for }} {title} $"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray],
                                              List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value
    and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's
        value and parameters at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    rec_values, rec_weights = [], []

    def gd_state_recorder_callback(weights, val, **kwargs):
        rec_values.append(val)
        rec_weights.append(weights)

    return gd_state_recorder_callback, rec_values, rec_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    from itertools import product
    names = {"L1": rf"\ell_{{1}} \text{{-norm}} = ||w||_{{1}}",
             "L2": rf"\ell_{{2}} \text{{-norm}} = ||w||^{{2}}_{{2}}"}
    out_type = "last"
    y_titles = [" ||w||_{1} ", " ||w||^{2}_{2} "]
    fig2a = go.Figure(
        layout=go.Layout(title=rf"$ \text{{The Convergence Rate of }} {y_titles[0]} $",
                         xaxis_title=r"$\text{GD iteration}$", yaxis_title=rf"${y_titles[0]}$",
                         legend=dict(x=0.85, y=0.9), width=1000))
    fig2a.update_xaxes(type="log")
    fig2b = go.Figure(
        layout=go.Layout(title=rf"$ \text{{The Convergence Rate of }} {y_titles[1]} $",
                         xaxis_title=r"$\text{GD iteration}$", yaxis_title=rf"${y_titles[1]}$",
                         legend=dict(x=0.85, y=0.85), width=1000))
    fig2b.update_xaxes(type="log")

    for k, lr in product([1, 2], etas):
        subtitle = rf"\text{{, fixed }} \eta={lr} \text{{, out_type}}={out_type}"
        f = L1 if k == 1 else L2
        gd_state_recorder_callback, rec_values, rec_weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=1000, out_type=out_type,
                             callback=gd_state_recorder_callback)
        w = gd.fit(f=f(deepcopy(init)), X=None, y=None)
        fig1 = plot_descent_path(module=L1 if k == 1 else L2, descent_path=np.array(rec_weights),
                                 title=names[f"L{k}"] + subtitle)
        fig1.write_image(f"./ex6.figs/plot.descent.path.L{k}"
                         f".fixed.eta({lr}).type.({out_type}).png")
        x = np.arange(1, gd.max_iter_ + 1)
        if k == 1:
            fig2a.add_traces(data=[go.Scatter(name=rf"$\eta={lr}$", x=x, y=rec_values,
                                              mode="markers" if lr == 1 else "lines")])
        if k == 2:
            fig2b.add_traces(data=[go.Scatter(name=rf"$\eta={lr}$", x=x, y=rec_values,
                                              mode="lines")])

        print(f"L{k} with eta=({lr}) and out_type=({out_type}) takes iter={len(rec_weights)}\n"
              f"the optimal w={w} and the lowest loss={f(w).compute_output():.3f}")

    fig2a.write_image(f"./ex6.figs/plot.convergence.rate.L1.fixed.rate"
                      f".out_type.({out_type}).png")
    fig2b.write_image(f"./ex6.figs/plot.convergence.rate.L2.fixed.rate"
                      f".out_type.type.({out_type}).png")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying
    # learning rate
    fig3 = go.Figure(layout=go.Layout(title=rf"$ \text{{The Convergence Rate of }} ||w||_{{1}}"
                                            rf"\text{{ with Exponential LR, }} \eta={eta}$",
                                      xaxis_title=r"$\text{GD iteration}$",
                                      yaxis_title=r"$ ||w||_{1} $",
                                      template="simple_white", width=1000,
                                      legend=dict(x=0.7, y=0.8)))
    fig3.update_xaxes(type="log")

    for d_rate in gammas:
        gd_state_recorder_callback, rec_values, rec_weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=d_rate),
                             max_iter=1000, callback=gd_state_recorder_callback)
        w = gd.fit(f=L1(deepcopy(init)), X=None, y=None)

        # Plot algorithm's convergence for the different values of gamma
        fig3.add_traces(data=[
            go.Scatter(name=rf"$\gamma={d_rate}$", mode="lines",
                       x=np.arange(1, gd.max_iter_ + 1), y=rec_values)])

        # Plot descent path for gamma=0.95
        if d_rate == 0.95:
            fig4 = plot_descent_path(module=L1, descent_path=np.array(rec_weights),
                                     title=rf"\ell_{{1}} \text{{-norm}} = ||w||_{{1}}"
                                           rf"\text{{, ExponentialLR }} \gamma={d_rate}")
            fig4.write_image(f"./ex6.figs/plot.descent.path.L1.ExponentialLR"
                             f"gamma({d_rate}).png")

    fig3.write_image(f"./ex6.figs/plot.convergence.rate.L1.ExponentialLR.png")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    # solver = GradientDescent(learning_rate=ExponentialLR(base_lr=0.01, decay_rate=0.99))
    # solver = GradientDescent(tol=1e-9, max_iter=int(1e5),
    #                          learning_rate=ExponentialLR(base_lr=1.0, decay_rate=0.998))
    solver = GradientDescent(learning_rate=FixedLR(base_lr=1e-4), max_iter=20000)
    model = LogisticRegression(solver=deepcopy(solver)).fit(X_train.to_numpy(),
                                                            y_train.to_numpy())
    y_prob = model.predict_proba(X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    log_reg = LogisticRegression(solver=deepcopy(solver),
                                 alpha=optimal_threshold).fit(X_train.to_numpy(),
                                                              y_train.to_numpy())
    test_error = log_reg.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f"Optimal Threshold value is: {optimal_threshold:.3f}")
    print(f"Test error using it = {test_error:.3f}")
    fig5 = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="",
                         showlegend=False, marker_size=5, marker_color="#795dba",
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{"
                                       "y:.3f}"),
              go.Scatter(name=r"$\text{Optimal Threshold Value}$",
                         x=[fpr[optimal_idx]], y=[tpr[optimal_idx]],
                         mode='markers+text', marker=dict(color='tomato', symbol="star", size=10),
                         text=[rf"$\alpha^{{*}}={optimal_threshold:.3f}$"],
                         textposition="top left")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"),
                         legend=dict(x=0.6, y=0.1)))
    fig5.write_image(f"./ex6.figs/plot.ROC.curve.png")

    # reg_errors = np.zeros((2, 7))
    # lams = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    # for i, lam in enumerate(lams):
    #     log_reg = LogisticRegression(solver=deepcopy(solver), penalty='l1', lam=lam,
    #                                  alpha=0.5).fit(X_train.to_numpy(), y_train.to_numpy())
    #     reg_errors[0][i] = log_reg.loss(X_test.to_numpy(), y_test.to_numpy())
    #
    #     log_reg = LogisticRegression(solver=deepcopy(solver), penalty='l2', lam=lam,
    #                                  alpha=0.5).fit(X_train.to_numpy(), y_train.to_numpy())
    #     reg_errors[1][i] = log_reg.loss(X_test.to_numpy(), y_test.to_numpy())
    #
    # print(reg_errors)
    # best_idx_l1 = np.argmin(reg_errors[0])
    # best_idx_l2 = np.argmin(reg_errors[1])
    # print(f"l1 best lam={lams[best_idx_l1]}   loss={reg_errors[0][best_idx_l1]}")
    # print(f"l2 best lam={lams[best_idx_l2]}   loss={reg_errors[1][best_idx_l2]}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to
    # specify values of regularization parameter
    # solver = GradientDescent(learning_rate=FixedLR(base_lr=1e-4), max_iter=2000)
    # n_evaluations = 200
    # lam_range = np.linspace(0.001, 1.5, num=n_evaluations)
    # L1_scores, L2_scores = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))
    lam_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    L1_scores, L2_scores = np.zeros((len(lam_range), 2)), np.zeros((len(lam_range), 2))
    # print("start")
    for i, lam in enumerate(lam_range):
        print(f"lambda={lam}")
        L1_scores[i] = cross_validate(LogisticRegression(solver=deepcopy(solver),
                                                         penalty='l1', lam=lam, alpha=0.5),
                                      X_train.to_numpy(), y_train.to_numpy(),
                                      misclassification_error)
        L2_scores[i] = cross_validate(LogisticRegression(solver=deepcopy(solver),
                                                         penalty='l2', lam=lam, alpha=0.5),
                                      X_train.to_numpy(), y_train.to_numpy(),
                                      misclassification_error)

    fig = make_subplots(2, 1,
                        subplot_titles=[r"$\ell_1\text{ Regularization}$",
                                        r"$\ell_2\text{ Regularization}$"],
                        shared_xaxes=True) \
        .update_layout(title=r"$\text{Train and Validation Errors (averaged over the k-folds)}$",
                       width=750, height=300) \
        .update_xaxes(title=r"$\lambda\text{ - Regularization parameter}$") \
        .add_traces([go.Scatter(x=lam_range, y=L1_scores[:, 0],
                                name=r"$\ell_1 \text{Train Error}$"),
                     go.Scatter(x=lam_range, y=L1_scores[:, 1],
                                name=r"$\ell_1 \text{Validation Error}$"),
                     go.Scatter(x=lam_range, y=L2_scores[:, 0],
                                name=r"$\ell_2 \text{Train Error}$"),
                     go.Scatter(x=lam_range, y=L2_scores[:, 1],
                                name=r"$\ell_2 \text{Validation Error}$")],
                    rows=[1, 1, 2, 2],
                    cols=[1, 1, 1, 1])
    fig.write_image("./ex6.figs/l1.l2.train.validation.errors.png")

    reg_L1 = lam_range[np.argmin(L1_scores[:, 1])]
    reg_L2 = lam_range[np.argmin(L2_scores[:, 1])]

    L1_log_reg_loss = LogisticRegression(solver=deepcopy(solver),
                                         penalty='l1', lam=reg_L1, alpha=0.5). \
        fit(X_train.to_numpy(), y_train.to_numpy()).loss(X_test.to_numpy(), y_test.to_numpy())
    L2_log_reg_loss = LogisticRegression(solver=deepcopy(solver),
                                         penalty='l2', lam=reg_L2, alpha=0.5). \
        fit(X_train.to_numpy(), y_train.to_numpy()).loss(X_test.to_numpy(), y_test.to_numpy())

    print(f"# ------------------------------------------------------------------------- #\n"
          f"#    Regularized Regression Fitting - Selecting Regularization Parameter    #\n"
          f"# ------------------------------------------------------------------------- #\n"
          f"\tCV-optimal Logistic l1-regularization parameter: {reg_L1}\n"
          f"\tCV-optimal Logistic l2-regularization parameter: {reg_L2}\n"
          f"\n\n"
          f"Test errors of models:\n"
          f"\tl1-regularized: {L1_log_reg_loss}\n"
          f"\tl2-regularized: {L2_log_reg_loss}\n"
          f"# ------------------------------------------------------------------------- #\n\n")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
