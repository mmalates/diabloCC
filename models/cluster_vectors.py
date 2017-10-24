import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
pd.get_option('display.width')
pd.options.display.width = 200


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


clusterdf = pd.read_csv('5ward.csv', parse_dates=[
    'date'], date_parser=dateparse)

meanvect = clusterdf.groupby(by='label').mean()
vectors = meanvect[['temp_high', 'temp_low', 'hum_avg',
                    'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg']].values
index = meanvect[['temp_high', 'temp_low', 'hum_avg',
                  'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg']].index


dic = defaultdict(list)
for label, values in zip(index, vectors):
    dic[label] = values

dic
