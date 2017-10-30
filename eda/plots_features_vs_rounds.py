import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/data_clean.csv', parse_dates=[
    'date'], date_parser=dateparse)


cols = ['rounds', 'temp_high', 'temp_low', 'dew_avg', 'hum_avg',
        'sea_avg', 'vis_low', 'wind_avg', 'prec', 'DOW', 'DOY']
for col1, col2 in combinations(cols, 2):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(df[col1].values, df[col2].values)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    plt.title(col2 + ' vs ' + col1)
    plt.savefig('plots/' + col1 + '_vs_' + col2 + '.png')
    plt.show()
