import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(wl=DecisionStump, iterations=n_learners).fit(train_X, train_y)
    train_errors, test_errors = [], []
    for T in range(1, n_learners + 1):
        train_errors.append(ada_boost.partial_loss(train_X, train_y, T))
        test_errors.append(ada_boost.partial_loss(test_X, test_y, T))

    fig = go.Figure(data=[go.Scatter(x=np.arange(n_learners + 1), y=train_errors,
                                     mode="lines", name="Train error"),
                          go.Scatter(x=np.arange(n_learners + 1), y=test_errors,
                                     mode="lines", name="Test error")],
                    layout=go.Layout(title="Training & Test Errors as a Function of Number of "
                                           "Fitted Learners",
                                     xaxis_title="Fitted Learners",
                                     yaxis_title="Error Values",
                                     width=1000))
    fig.write_image(f"./ex4.figs/q1.adaBoost.noise={noise}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} iterations" for t in T],
                        horizontal_spacing=0.05)
    fig.update_layout(title="Decision Boundaries With Different Number of Iterations "
                            "without noise\n\n\n",
                      font=dict(family="sans serif", size=17), width=850, showlegend=False)

    colors = np.array(['#2596be', '#e28743'])
    symbols = np.array(['star', 'circle'])
    for t in range(len(T)):
        fig.add_traces(data=[go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                        marker={"color": test_y},
                                        # marker=dict(color=colors[np.where(test_y == 1, 1, 0)],
                                        #             symbol=symbols[np.where(test_y == 1, 1, 0)]),
                                        showlegend=False),
                             decision_surface(lambda X: ada_boost.partial_predict(X, T[t]),
                                              lims[0], lims[1], showscale=False)],
                       rows=t // 2 + 1, cols=t % 2 + 1)
    fig.write_image(f"./ex4.figs/q2.decision.surfaces.noise={noise}.png")

    # Question 3: Decision surface of best performing ensemble
    lowest = np.argmin(test_errors)
    accu = accuracy(test_y, ada_boost.partial_predict(test_X, int(lowest)))
    lowest_h = lambda X: ada_boost.partial_predict(X, int(lowest))

    fig = go.Figure(data=[go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                     marker={"color": test_y},
                                     # marker=dict(color=colors[np.where(test_y == 1, 1, 0)],
                                     #             symbol=symbols[np.where(test_y == 1, 1, 0)]),
                                     showlegend=False),
                          decision_surface(lowest_h, lims[0], lims[1], showscale=False)],
                    layout=go.Layout(title=f"Decision Surface of the Ensemble of size = {lowest} "
                                           f"with accuracy = {accu}",
                                     yaxis={'visible': False, 'showticklabels': False},
                                     xaxis={'visible': False, 'showticklabels': False},
                                     width=1000))
    fig.write_image(f"./ex4.figs/q3.decision.boundary.noise={noise}.png")

    # Question 4: Decision surface with weighted samples
    cur_D = (ada_boost.D_ / np.max(ada_boost.D_)) * 15
    fig = go.Figure(data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                                     marker={"color": train_y,
                                             "colorscale": [custom[0], custom[-1]],
                                             "size": cur_D,
                                             "line": {"color": "black", "width": 0.2}},
                                     showlegend=False),
                          decision_surface(ada_boost.predict, lims[0], lims[1], showscale=False)],
                    layout=go.Layout(title=f"Decision Surface with Weighted Full Ensemble",
                                     yaxis={'visible': False, 'showticklabels': False},
                                     xaxis={'visible': False, 'showticklabels': False},
                                     width=1000))
    fig.write_image(f"./ex4.figs/q4.full.decision.surfaces.noise={noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
