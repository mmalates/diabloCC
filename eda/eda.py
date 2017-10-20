import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose

# read data to pandas dataframe


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


data = pd.read_csv('diablo.csv', parse_dates=['date'], date_parser=dateparse)


# make autocorrelation plot
# autocorrelation_plot(series)
# plt.ylim([-0.25,0.5])
# plt.show()

# make seasonal decomposition
def decompose(series, filename):
    decomposition = seasonal_decompose(series, model='additive', freq=260)
    fig, ax = plt.subplots(figsize=(20, 20))
    fig = decomposition.plot()
    plt.savefig(filename + '_decomposition')


# get rounds as series with date as index
all = pd.Series(data.rounds.values, index=data.date)
# get series cuts for weekday/weekend and days of the week
weekend = data[data['DOW'] >= 5]
weekday = data[data['DOW'] < 5]
tuesday = data[data['DOW'] == 1]
wednesday = data[data['DOW'] == 2]
thursday = data[data['DOW'] == 3]
friday = data[data['DOW'] == 4]

# make datetime series


def datetime_series(df):
    series = pd.Series(df.rounds.values, index=df.date)
    return series


weekend = datetime_series(weekend)
weekday = datetime_series(weekday)
tuesday = datetime_series(tuesday)
wednesday = datetime_series(wednesday)
thursday = datetime_series(thursday)
friday = datetime_series(friday)


decompose(all, 'all')
decompose(weekend, 'weekend')
decompose(weekday, 'weekday')
decompose(tuesday, 'tuesday')
decompose(wednesday, 'wednesday')
decompose(thursday, 'thursday')
decompose(friday, 'friday')
