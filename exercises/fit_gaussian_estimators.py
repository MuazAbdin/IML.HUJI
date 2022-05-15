from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from utils import *

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    estimator = UnivariateGaussian()
    # Question 1 - Draw samples and print fitted model
    mu, var, m_size = 10, 1, 1000
    samples = np.random.normal(mu, var, size=m_size)
    estimator.fit(samples)
    expectation, variance = estimator.mu_, estimator.var_
    print(expectation, variance)

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.linspace(10, m_size, 100).astype(np.int32)
    estimated_mean = []
    abs_dist = []
    for size in sample_sizes:
        estimated_mean += [estimator.fit(samples[:size]).mu_]
        abs_dist += [abs(estimator.fit(samples[:size]).mu_ - mu)]

    go.Figure([go.Scatter(x=sample_sizes, y=abs_dist, mode='lines', name=r'distance$')],
              layout=go.Layout(
                  title="$\\text{Absolute Distance between }\mu\\text{ and }\widehat\mu$",
                  xaxis_title="$\\text{number of samples}$",
                  yaxis_title="r$\\text{distance}$",
                  height=500)).show()

    go.Figure([go.Scatter(x=sample_sizes, y=estimated_mean, mode='markers+lines',
                          name=r'$\widehat\mu_1$'),
               go.Scatter(x=sample_sizes, y=[mu] * len(sample_sizes), mode='lines', name=r'$\mu$')],
              layout=go.Layout(
                  title=r"$\text{Estimation of Expectation As Function Of Sample Size}$",
                  xaxis_title="$\\text{number of samples}$",
                  yaxis_title="r$\hat\mu$",
                  height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sample_values = np.array([])
    pdf_values = np.array([])
    for idx in range(len(sample_sizes)):
        sample = samples[:sample_sizes[idx]]
        sample_values = np.hstack((sample_values, sample))
        pdf_values = np.hstack((pdf_values, estimator.pdf(sample)))

    go.Figure([go.Scatter(x=sample_values, y=pdf_values, mode='markers',
                          marker=dict(color="rgba(244,175,97,255)"),
                          name=r'$Empirical PDF$')],
              layout=go.Layout(
                  title=r"$\text{Empirical PDFs of samples ~ N(10,1)}$",
                  xaxis_title="$\\text{value of sample}$",
                  yaxis_title="$\\text{density}$",
                  height=500)).show()

    # # Likelihood evaluation (slide #1 p.24)
    # mu_1 = UnivariateGaussian.log_likelihood(mu=0, sigma=1, X=np.array([-1, 0, 0, 1]))
    # mu_2 = UnivariateGaussian.log_likelihood(mu=1, sigma=1, X=np.array([-1, 0, 0, 1]))
    # print(mu_1, mu_2)


def test_multivariate_gaussian():
    estimator = MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]]
    s_size = 1000
    samples = np.random.multivariate_normal(mu, cov, size=s_size)
    estimator.fit(samples)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    r = 200
    f = np.linspace(-10, 10, r)
    models = np.transpose(np.vstack([np.tile(f, r), np.zeros(r * r),
                                     np.repeat(f, r), np.zeros(r * r)]))
    lls = []
    for model in models:
        lls += [MultivariateGaussian.log_likelihood(mu=model, cov=np.array(cov), X=samples)]

    go.Figure(data=go.Heatmap(z=lls, x=np.tile(f, r), y=np.repeat(f, r)),
              layout=go.Layout(
                  title="$\\text{Multivariate Gaussian Log Likelihood}$",
                  xaxis_title="r$\\text{values of f}\f_1$",
                  yaxis_title="r$\\text{values of f}\f_2$",
                  height=650, width=650)).show()

    # Question 6 - Maximum likelihood
    idx = np.argmax(lls)
    print(f"The maximum log-likelihood value is {lls[idx]:.3f}\n"
          f"with the model: mu = {models[idx]}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

    # X = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10,
    #               4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4,
    #               1, 2, 1, -4, -4, 1, 3, 2, 6, -6, 8, 3,
    #               -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1,
    #               0, 3, 5, 0, -2])
    #
    # print(UnivariateGaussian.log_likelihood(1, 1, X))
    # print(UnivariateGaussian.log_likelihood(10, 1, X))
