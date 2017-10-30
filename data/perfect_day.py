import pandas as pd
import numpy as np


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


data = pd.read_csv('data_notes.csv', parse_dates=[
                   'date'], date_parser=dateparse)

for item in data.Notes.values:
    print item

mask_perfect = []

for row in data['Notes'].values:
    mask_perfect.append('perfect' in row.lower())

sum(mask_perfect)

cluster_features = ['temp_high', 'temp_low',
                    'hum_avg', 'wind_avg', 'prec', 'vis_avg']
perfectdf = data[mask_perfect]

perfect_weather = perfectdf[cluster_features].describe().T['mean'].values

print perfect_weather
