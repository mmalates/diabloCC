import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/data_clean.csv', parse_dates=[
    'date'], date_parser=dateparse)


# cols = ['temp_high', 'temp_avg', 'temp_low', 'dew_high', 'dew_avg', 'dew_low', 'hum_high', 'hum_avg', 'hum_low',
#         'sea_high', 'sea_avg', 'sea_low', 'vis_high', 'vis_avg', 'vis_low', 'wind_high', 'wind_avg', 'wind_low', 'prec', 'DOY']

cols = ['rounds', 'temp_high', 'temp_low', 'dew_avg', 'hum_avg',
        'sea_avg', 'vis_low', 'wind_avg', 'prec', 'DOW', 'DOY']
for col in cols:
    for col2 in cols:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.scatter(df[col].values, df[col2].values)
        plt.title(col + ' vs ' + col2)
        plt.savefig('scatter/+' + col + '_vs_' + col2 + '.png')
        plt.show()
