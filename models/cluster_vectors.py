import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
pd.get_option('display.width')
pd.options.display.width = 200

# test clustering accuracy for new points


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


# read data with cluster labels
clusterdf = pd.read_csv('5ward.csv', parse_dates=[
    'date'], date_parser=dateparse)


cluster_features = ['temp_high', 'temp_low',
                    'hum_avg', 'wind_avg', 'prec', 'vis_avg']
# find the centroid of each cluster
meanvect = clusterdf.groupby(by='label').mean()
vectors = meanvect[cluster_features].values
index = meanvect[cluster_features].index

# dictionary for storing centroid vectors
dic = defaultdict(list)

for centroid, vector in zip(index, vectors):
    dic[centroid] = vector

scale = np.array([0.00505801,  0.00842202,  0.00908669,
                  0.04073148,  0.13243243, 0.09606463])
labels = []
for day in clusterdf[cluster_features].values:
    diffs = defaultdict(float)
    for label, vector in dic.iteritems():
        # print (scale * day) - (scale * vector)
        diffs[label] = np.sum(np.abs(np.subtract(scale * day, scale * vector)))
    labels.append(min(diffs.items(), key=lambda x: x[1]))
label_col = []
index_dic = {5.0: 10.0,
             4.0: 6.0,
             3.0: 5.0,
             2.0: 8.0,
             1.0: 7.0}
for cluster_label in labels:
    label_col.append(cluster_label[0])
clusterdf['pred_labels'] = np.array(label_col)
clusterdf.head()
clusterdf['label_diff'] = np.abs(clusterdf.pred_labels - clusterdf.label)
# print clusterdf[['label', 'label_diff']]
