import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.model_selection import KFold


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/train_nznt_cum.csv',
                 parse_dates=['date'], date_parser=dateparse)


def datetime_series(df, col):
    '''
    Produces a datetime series of a specified column in a dataframe

    Input:
        df : dataframe
        col : column in dataframe to prod

    Output:
        Pandas Datetime Series
    '''
    series = pd.Series(df[col].values.astype(
        float), index=pd.DatetimeIndex(df.date))
    return series


# decompose rounds
rounds_series = datetime_series(df, 'rounds')
rounds_decomposition = seasonal_decompose(
    rounds_series.values, model='additive', freq=305)
rounds_residual = rounds_decomposition.observed - rounds_decomposition.seasonal


kf = KFold(n_splits=10, shuffle=True)
error = []
y = rounds_residual

for train_index, test_index in kf.split(y):
    y_train, y_test = y[train_index], y[test_index]
    predictions = np.ones(len(y_test)) * np.mean(y_train)
    error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))

print np.mean(error)
