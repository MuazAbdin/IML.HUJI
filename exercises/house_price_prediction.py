from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Tuple
from scipy import stats
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

IRRELEVANT = ['id', 'date', 'lat', 'long']
POSITIVE = ['price', 'sqft_living', 'sqft_lot', 'floors', 'yr_built',
            'sqft_living15', 'sqft_lot15']
NON_NEGATIVE = ['bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement', 'yr_renovated']


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # drop rows with NaN values, duplicate rows, and IRRELEVANT columns
    df = pd.read_csv(filename).dropna().drop_duplicates().drop(columns=IRRELEVANT)

    # drop invalid feature values
    for feature in POSITIVE:
        df = df[df[feature] > 0]
    for feature in NON_NEGATIVE:
        df = df[df[feature] >= 0]

    # remove the outliers of some features
    # df = df[
    #     (np.abs(stats.zscore(df.select_dtypes(include=np.number))) < 3).all(axis=1)
    # ]

    # converted renovation feature from categorical to numerical binary feature
    df['renovated'] = np.where(df['yr_renovated'] >= np.percentile(df.yr_renovated.unique(), 70),
                               1, 0)
    df = df.drop(columns=['yr_renovated'])

    # remove the outliers of some features
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < int(1e6)]
    df = df[df['sqft_basement'] < 3500]
    df = df[df["sqft_lot15"] < int(5e5)]

    # One-Hot encoding for categorical data
    df["zipcode"] = df["zipcode"].astype(np.int32)
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])

    return df.drop(columns=['price']), df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False) |
    #                X.columns.str.contains('^built_time_', case=False))]
    for f in X:
        pearson_rho = np.cov(X[f], y)[1, 0] / (np.std(X[f]) * np.std(y))

        fig = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y",
                         title=f"Correlation Between {f} Values and Response <br>Pearson "
                               f"Correlation = {pearson_rho:.3f}",
                         labels={"x": f"{f} Values", "y": "Response Values"})
        fig.write_image(f"{output_path}/pearson.correlation.{f}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    # X, y = load_data("house_prices.csv")
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size
    # (mean-2*std, mean+2*std)
    # lre = LinearRegression()
    results = np.zeros((91, 2))
    for p in range(10, 101):
        loss = []
        for i in range(10):
            lre = LinearRegression()
            X_p = train_X.sample(frac=p / 100)
            y_p = train_y.reindex_like(X_p)
            lre.fit(X_p.to_numpy(), y_p.to_numpy())
            loss += [lre.loss(test_X.to_numpy(), test_y.to_numpy())]
            if p == 100:
                break
        results[p - 10, :] = np.array([np.mean(loss), np.var(loss)])

    ps = np.linspace(10, 100, 91).astype(np.int32)
    means, stds = results[:, 0], np.sqrt(results[:, 1])

    fig = go.Figure([go.Scatter(x=ps, y=means - 2 * stds, fill=None, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=ps, y=means + 2 * stds, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=ps, y=means, mode="markers+lines", name="Mean loss",
                                marker=dict(color="black", size=1), showlegend=True)],
                    layout=go.Layout(
                        title=r"$\text{The mean loss as a function of p%}$",
                        xaxis_title="Percentage of training set",
                        yaxis_title="Mean loss",
                        height=500))
    fig.write_image("mean.loss.over.training.percentage.png")
