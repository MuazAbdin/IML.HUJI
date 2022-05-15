from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first 2 columns represent
    features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly
    separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, labels = load_dataset(filename=f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(fit: Perceptron, x: np.ndarray, y: int):
            fit.coefs_ += x * y
            losses.append(fit.loss(X, labels))

        Perceptron(callback=loss_callback).fit(X, labels)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(
            [go.Scatter(x=np.linspace(0, len(losses)), y=losses, mode='lines',
                        marker=dict(size=15, color='#DE9B25'))],
            layout=go.Layout(title=f"The Loss Over The {n} Training Set",
                             font=dict(
                                 family="sans serif",
                                 size=18
                             ),
                             xaxis=dict(title="Fitting Iteration"),
                             yaxis=dict(title="Loss Over Training Set"),
                             showlegend=False))
        fig.write_image(f"./ex3.figs/loss.over."
                        f"{'.'.join(f'{n}'.split(' '))}.training.set.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker=dict(color='black'))


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(filename=f'../datasets/{f}')

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        lda_res = lda.predict(X)
        gnb = GaussianNaiveBayes().fit(X, y)
        gnb_res = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on
        # the left and LDA predictions on the right. Plot title should specify dataset used and
        # subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        model_names = [f'Gaussian Naive Bayes (acc={accuracy(y, gnb_res):.3f})',
                       f'LDA (acc={accuracy(y, lda_res):.3f})']
        fig = make_subplots(rows=1, cols=2, subplot_titles=model_names, horizontal_spacing=0.05)
        fig.update_layout(title=f"The Responses over the ({f.split('.')[0]}) Set",
                          font=dict(family="sans serif", size=18), width=1200, height=450,
                          showlegend=False)

        # Add traces for data-points setting symbols and colors
        colors = np.array(['#2596be', '#e28743', '#063970'])
        symbols = np.array(['star', 'circle', 'triangle-up'])
        fig.add_traces(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                        marker=dict(color=colors[gnb_res],
                                                    symbol=symbols[y])),
                             go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                        marker=dict(color=colors[lda_res],
                                                    symbol=symbols[y]))],
                       rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces(data=[go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode='markers',
                                        marker=dict(color='black', symbol='x', size=10)),
                             go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode='markers',
                                        marker=dict(color='black', symbol='x', size=10))],
                       rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(3):
            fig.add_traces(data=[get_ellipse(gnb.mu_[k], np.diag(gnb.vars_[k])),
                                 get_ellipse(lda.mu_[k], lda.cov_)],
                           rows=[1, 1], cols=[1, 2])

        fig.write_image(f"./ex3.figs/gnb.vs.lda.{f.split('.')[0]}.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
