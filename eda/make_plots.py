import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from statsmodels.tsa.seasonal import seasonal_decompose
# %matplotlib inline

# read data from csv


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/data.csv',
                 parse_dates=['date'], date_parser=dateparse)

# makes a scatter plot and labels axes


def plot_data(x, y, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(40, 6))
    ax.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# makes a line plot and labels axes


def plot_line(x, y, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(40, 6))
    ax.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# make DOW cuts
weekend = df['DOW'] >= 5
weekday = df['DOW'] < 5
tuesday = df['DOW'] == 1
wednesday = df['DOW'] == 2
thursday = df['DOW'] == 3
friday = df['DOW'] == 4

days = df.date.values
rounds = df.rounds.values
no_zero = rounds > 0

days_no_zero = days[no_zero]
rounds_no_zero = rounds[no_zero]

walkers = df.walkers.values
riders = rounds - walkers
tourneys = df.tourney.values
round_minus_tourney = rounds - tourneys
no_zero_or_tourney = round_minus_tourney > 0

df_no_zero_or_tourney = df[no_zero_or_tourney]

# plot rounds, walkers, and riders for several different DOW cuts
plot_data(days[no_zero], rounds[no_zero], 'date', 'rounds')
plot_data(days[no_zero], riders[no_zero], 'date', 'riders')
plot_data(days[no_zero], walkers[no_zero], 'date', 'walkers')
plot_data(days[no_zero_or_tourney],
          round_minus_tourney[no_zero_or_tourney], 'date', 'rounds_no_touney')
plot_data(days[no_zero_or_tourney], walkers[no_zero_or_tourney],
          'date', 'walker_no_tourney')
plot_line(days[weekend], rounds[weekend], 'date', 'weekend rounds')
plot_line(days_no_zero[weekday[no_zero]], rounds_no_zero[weekday[no_zero]],
          'date', 'weekday rounds')
plot_line(days[tuesday], rounds[tuesday], 'date', 'tuesday rounds')
plot_line(days[wednesday], rounds[wednesday], 'date', 'wednesday rounds')
plot_line(days[thursday], rounds[thursday], 'date', 'thursday rounds')
plot_line(days[friday], rounds[friday], 'date', 'friday rounds')

# plt.show()

# replace T with a trace amount
df.loc[df.prec == 'T', 'prec'] = 0.001
df.prec = df.prec.astype(float)
df_no_zero_or_tourney.loc[df_no_zero_or_tourney.prec == 'T', 'prec'] = 0.001
df = df.fillna(0)
df_no_zero_or_tourney = df_no_zero_or_tourney.fillna(0)
# make a datetime series for decomposition of a given column

# print cleaned dataframe to file
df_no_zero_or_tourney.to_csv('../data/diablo_nznt.csv')


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
rounds_series = datetime_series(df_no_zero_or_tourney, 'rounds')
rounds_decomposition = seasonal_decompose(
    rounds_series.values, model='additive', freq=305)
rounds_residual = rounds_decomposition.observed - rounds_decomposition.seasonal

# fig, ax = plt.subplots(2, sharex=True, figsize=(40, 6))
# ax[0].plot(df.date.values, df.rounds.values)
# ax[1].plot(df.date.values, df.prec.values)
# plt.savefig('rounds_and_rain')

# scale precipitation so we can see variation compared to rounds
scale = np.ones(len(df_no_zero_or_tourney.date.values))
offset = np.ones(len(df_no_zero_or_tourney.date.values))
prec_arr = df_no_zero_or_tourney.prec.values
prec_scale = np.ones(len(prec_arr))
prec_scale *= 100
prec_offset = np.ones(len(prec_arr))
prec_offset *= 200
prec_arr = prec_arr.astype(float) * prec_scale
prec_arr += prec_offset

# plot decomposed rounds and precip
fig, ax = plt.subplots(figsize=(40, 6))
ax.plot(df_no_zero_or_tourney.date.values, prec_arr, label='precipitation')
ax.plot(df_no_zero_or_tourney.date.values, rounds_residual, label='rounds')
plt.legend()
plt.savefig('../plots/rounds_nznt_resid_and_precip')

# decompose wind
wind_series = datetime_series(df_no_zero_or_tourney, 'wind_avg')
wind_decomposed = seasonal_decompose(
    wind_series, model='additive', freq=305)
wind_residual = wind_decomposed.observed - wind_decomposed.seasonal
wind_residual = 10 * scale * wind_residual + 150 * offset
# scale10 = scale / 10.
# wind = df.wind_avg.values
# wind = wind.astype(float)* scale10
# wind += scale
fig, ax = plt.subplots(figsize=(40, 6))
ax.plot(df_no_zero_or_tourney.date.values, wind_residual, label='wind')
ax.plot(df_no_zero_or_tourney.date.values, rounds_residual, label='rounds')
plt.legend()
plt.savefig('../plots/rounds_nznt_resid_and_wind_resid')


# decompose temp
temp_series = datetime_series(df_no_zero_or_tourney, 'temp_avg')
temp_decomposed = seasonal_decompose(
    temp_series, model='additive', freq=305)
temp_residual = temp_decomposed.observed - temp_decomposed.seasonal
temp_residual = 5 * scale * temp_residual - 100 * offset

# temp = df.temp_avg.values
# temp = temp.astype(float)
# temp += scale
fig, ax = plt.subplots(figsize=(40, 6))
ax.plot(df_no_zero_or_tourney.date.values, temp_residual, label='temperature')
ax.plot(df_no_zero_or_tourney.date.values, rounds_residual, label='rounds')
plt.legend()
plt.savefig('../plots/rounds_nznt_resid_and_temp_resid')

# plots histogram of spread in residual rounds
pd.Series(rounds_residual).hist(bins=20)
plt.show()

# calculate mean and std of residual rounds to determine if a datapoint is a significant event
residual_mean = np.mean(rounds_residual)
residual_std = np.std(rounds_residual)
high_resid_mask = rounds_residual > residual_mean + 1.5 * residual_std
low_resid_mask = rounds_residual < residual_mean - 1.5 * residual_std


vlines_low = df_no_zero_or_tourney.date.values[low_resid_mask]
vlines_high = df_no_zero_or_tourney.date.values[high_resid_mask]


temp_residual += 150 * offset
wind_residual += 250 * offset
fig, ax = plt.subplots(figsize=(40, 6))
ax.plot(df_no_zero_or_tourney.date.values, wind_residual, label='wind')
ax.plot(df_no_zero_or_tourney.date.values, temp_residual, label='temperature')
ax.plot(df_no_zero_or_tourney.date.values, prec_arr, label='precipitation')
ax.vlines(vlines_low, 0, 600, color='c',
          linestyle='dashed', label='low rounds')
ax.vlines(vlines_high, 0, 600, color='g',
          linestyle='dashed', label='high rounds')
ax.plot(df_no_zero_or_tourney.date.values, rounds_residual, label='rounds')
plt.legend()
plt.savefig('../plots/rounds_nznt_resid_and_all_resid')

# def plot_histgram(series):
#     spread = df['rounds']
