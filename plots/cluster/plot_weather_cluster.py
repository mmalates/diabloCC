import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


clusterdf = pd.read_csv('../../models/5ward.csv', parse_dates=[
    'date'], date_parser=dateparse)


def make_cluster_plots(df, col1, col2):
    seaborn.set(style='ticks')
    fg = seaborn.FacetGrid(data=df[[col1, col2, 'label']],
                           hue='label', hue_order=np.arange(5) + 1, aspect=2, size=8)
    fg.map(plt.scatter, col1, col2).add_legend()

    plt.savefig('5ward/' + str(col1) + '_' + str(col2) + '_5ward' + '.png')
    plt.show()


cols = ['temp_high', 'temp_low', 'hum_avg',
        'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg']

for col1 in cols:
    for col2 in cols:
        if col1 != col2:
            make_cluster_plots(clusterdf, col1, col2)
