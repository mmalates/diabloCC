import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn import preprocessing
import seaborn


pd.get_option('display.width')
pd.options.display.width = 200


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/data_clean.csv', parse_dates=[
    'date'], date_parser=dateparse)


mask = df.rounds < 200
sum(mask)
df = df[mask]


link_features = ['DOY', 'DOW', 'temp_high',
                 'temp_low', 'hum_avg', 'wind_avg', 'prec']

X = df.drop(['date', 'rounds', 'walkers'], axis=1)
x = X.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled, columns=X.columns)
X.head()


# links = ['ward', 'single', 'centroid',
#  'complete', 'average', 'weighted', 'median']
# create linkage for hierarchical clustering
links = ['ward', 'complete', 'weighted']
# create linkage for hierarchical clustering


def try_linkages(linkage_type):
    print linkage_type
    Z = linkage(X[link_features], linkage_type)
    c, coph_dists = cophenet(Z, pdist(X))

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z, truncate_mode='lastp', p=8, show_leaf_counts=False,
               show_contracted=True, leaf_rotation=90., leaf_font_size=8.)
    plt.hlines(100, 0, 100, 'k')
    plt.show()

    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()

    k = acceleration_rev.argmax() + 2
    print "clusters:", k
    for k in xrange(3):
        k += 3
        make_cluster_plots(k, linkage_type, Z)


def make_cluster_plots(k, linkage_type, Z):
    labels = fcluster(Z, k, criterion='maxclust')
    label_s = pd.Series(labels)
    print k
    print label_s.groupby(label_s).count()

    df['label'] = label_s
    seaborn.set(style='ticks')
    fg = seaborn.FacetGrid(data=df[['rounds', 'date', 'label']],
                           hue='label', hue_order=np.arange(k) + 1, aspect=3, size=8)
    fg.map(plt.scatter, 'date', 'rounds').add_legend()

    plt.savefig('../plots/' + str(k) + linkage_type + '_labeled_rounds.png')

    df_comp = pd.DataFrame()

    def describe_label(l):
        mask = df.label == l
        df_comp[i] = df[mask].describe().T['mean']

    for i in xrange(k):
        i += 1
        describe_label(i)

    fig, ax = plt.subplots(24, 1, figsize=(6, 40))
    for i, col in enumerate(df_comp.T.columns):
        ax[i].plot(df_comp.T[col])
        ax[i].set_ylabel(col)
    plt.savefig('../plots/' + str(k) + linkage_type + '_cluster_means.png')
    df.to_csv(str(k) + linkage_type + '.csv')


for link in links:
    try_linkages(link)
