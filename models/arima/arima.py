import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
%matplotlib inline
rcParams['figure.figsize'] = 15, 6
pd.options.display.max_rows = None


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


def load_datetime_series(filename, target):
    # load data
    data = pd.read_csv(filename, parse_dates=['date'], date_parser=dateparse)

    data = data.drop_duplicates()
    data['year_month'] = data.date.dt.strftime('%Y-%m')

    data = data.groupby(by='year_month').sum()

    # make month_year the index for datetime series creation
    data.index = pd.to_datetime(data.index.astype(str), format='%Y-%m')

    # sort by date
    data = data.sort_index()

    # get datetime series
    return data[target]


def fill_in_missing_months(series):
    mask = [False if x.split(
        '-')[1] != '07' else True for x in series.index.astype(str)]
    series[pd.Timestamp('2012-07-01')] = np.mean(series[mask])
    series = series.sort_index()
    mask = [False if x.split(
        '-')[1] != '01' else True for x in series.index.astype(str)]
    series[pd.Timestamp('2015-01-01')] = np.mean(series[mask])
    series = series.sort_index()
    return series

# create a differenced series


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# invert differenced value


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# correct for rolling mean


def subtract_rolling_mean(series):
    moving_avg = pd.rolling_mean(series, 12, center=True)
    series - moving_avg
    series.dropna(inplace=True)
    return series


if __name__ == '__main__':
    filepath = '../../data/data_clean.csv'
    target = 'rounds'
    series = load_datetime_series(filepath, target)
    series = fill_in_missing_months(series)
    series = subtract_rolling_mean(series)
    X = series.values
    X = X.astype('float32')
    history = [x for x in series[-24:]]
    predictions = list()
    forecast_months = 12
    prev_year = series[-12:]
    for i in range(forecast_months):
        # difference data
        months_in_year = 12
        diff = difference(history, months_in_year)
        # predict
        model = ARIMA(diff, order=(1, 1, 0))
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = inverse_difference(history, yhat, months_in_year)
        predictions.append(yhat)
        obs = prev_year[i]
        history.append(obs)
        print('>Predicted=%.3f' % yhat)
    total = sum(predictions)
    print('total rounds: {}').format(total[0])
