import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose(series, filename):
    '''
    Plots seasonally decomposed trend.  Series must have a DatetimeIndex
    Input:
        series with DatetimeIndex
        filename
    Output:
        None
        Saves figure to filename
    '''
    decomposition = seasonal_decompose(series, model='additive', freq=365)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.tick_params(labelsize=15)
    ax = decomposition.trend.plot()
    ax.set_xlabel('Date', size=30)
    ax.set_ylabel('Trend', size=30)
    ax.set_ylim([0, 100])
    plt.savefig(filename)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.tick_params(labelsize=15)
    ax = decomposition.observed.plot()
    ax.set_xlabel('Date', size=30)
    ax.set_ylabel('Observed', size=30)
    ax.set_ylim([0, 200])
    plt.savefig('observed.png')
    plt.show()


if __name__ == '__main__':
    # read data to pandas dataframe
    def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')

    data = pd.read_csv('../data/data_cum_clean.csv', parse_dates=[
                       'date'], date_parser=dateparse)

    # get rounds as series with date as index
    all = pd.Series(data.rounds.values, index=data.date)

    # fill in missing days with day before
    allf = all[~all.index.duplicated(keep='first')]
    allf = allf.asfreq('D')
    out = []
    for i, item in enumerate(allf.values):
        if np.isnan(item):
            out.append(allf.iloc[i - 2])
        else:
            out.append(item)

    # fill in big missing areas with average values
    avg = np.mean(pd.Series(np.array(out)).dropna())
    filled = pd.Series(np.array(out), index=allf.index).fillna(avg)

    # run seasonal decomposition and plot trend
    decompose(filled, 'trend')
