import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from scipy import stats

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    dates = pd.DatetimeIndex(df['Date'])
    df = df[df['Month'].isin(range(1, 13)) &
            df['Day'].isin(range(1, 32)) &
            (dates.year == df['Year']) &
            (dates.month == df['Month']) &
            (dates.day == df['Day'])]
    # Eliminate the outliers
    df = df[
        (np.abs(stats.zscore(df.select_dtypes(include=np.number))) < 3).all(axis=1)
    ]
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")
    # X.info()

    # Question 2 - Exploring data for specific country
    X_IL = X[X['Country'] == 'Israel']
    assert X_IL['Country'].unique() == 'Israel'
    X_IL = X_IL[X_IL["Temp"] > -50]
    X_IL['Year'] = X_IL['Year'].astype(str)
    fig = px.scatter(X_IL, x="DayOfYear", y="Temp", color="Year",
                     title="The Average Daily Temperature in Israel")
    fig.write_image("average.temperature.israel.png")

    df_std_temp = X_IL.groupby(['Month'])['Temp'].std().reset_index(name='STD')
    # print(df_std_temp)
    fig_1 = px.bar(df_std_temp, x="Month", y="STD",
                   title="The Standard Deviation of the Daily Temperatures")
    fig_1.write_image("std.temperature.israel.png")

    # Question 3 - Exploring differences between countries
    df_diff = X.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']})
    df_mean = X.groupby(['Country', 'Month'])['Temp'].mean().reset_index(name='mean')
    df_std = X.groupby(['Country', 'Month'])['Temp'].std().reset_index(name='STD')
    fig_2 = px.line(df_mean, x="Month", y="mean", color="Country", error_y=df_std['STD'],
                    title="The Average Monthly Temperature in Different Countries")
    fig_2.write_image("average.monthly.temperature.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(X_IL.DayOfYear.to_frame(), X_IL.Temp)
    train_X, test_X = train_X.DayOfYear, test_X.DayOfYear

    loss_dict = {'k': [], 'loss': []}
    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(train_X.to_numpy(), train_y.to_numpy())
        loss_dict['k'] += [k]
        loss_dict['loss'] += [round(pf.loss(test_X.to_numpy(), test_y.to_numpy()), ndigits=2)]

    df_loss = pd.DataFrame(data=loss_dict)
    print(df_loss)
    fig_3 = px.bar(df_loss, x="k", y="loss", text="loss",
                   title="The Test Error Recorded for Each Value of k")
    fig_3.write_image("test.error.k.values.png")

    # Question 5 - Evaluating fitted model on different countries
    pfm = PolynomialFitting(k=4)
    pfm.fit(X_IL.DayOfYear, X_IL.Temp)
    test_sets = X['Country'].unique()
    test_sets = test_sets[test_sets != 'Israel']
    test_sets = [X[X['Country'] == c] for c in test_sets]

    error = {'country': [], 'value': []}
    for test_set in test_sets:
        error['country'] += [test_set['Country'].iloc[0]]
        error['value'] += [round(pfm.loss(test_set.DayOfYear, test_set.Temp), ndigits=2)]

    df_error = pd.DataFrame(data=error)
    print(df_error)
    fig_4 = px.bar(df_error, x="country", y="value", text="value",
                   title="The Error of the Model Trained on Israel Records")
    fig_4.write_image("model.error.png")
