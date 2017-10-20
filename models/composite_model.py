from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_rows = 2000


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/5ward.csv',
                 parse_dates=['date'], date_parser=dateparse)

df = df[df.rounds < 200]
features = ['DOY', 'DOW', 'temp_high', 'temp_low',
            'hum_avg', 'wind_avg', 'prec', 'label']

X_train1, x_test1, y_train1, y_test1 = train_test_split(
    df[features], df['rounds'])


mask1 = X_train1.label == 1
mask2 = X_train1.label == 2
mask3 = X_train1.label == 3
mask4 = X_train1.label == 4
mask5 = X_train1.label == 5


X1 = X_train1[mask1]
X2 = X_train1[mask2]
X3 = X_train1[mask3]
X4 = X_train1[mask4]
X5 = X_train1[mask5]


y1 = y_train1[mask1]
y2 = y_train1[mask2]
y3 = y_train1[mask3]
y4 = y_train1[mask4]
y5 = y_train1[mask5]

Xs = [X1, X2, X3, X4, X5]
ys = [y1, y2, y3, y4, y5]

grid = []


def RF_model():
    print "setting up the model"
    rf = RandomForestRegressor(n_estimators=50, bootstrap=True,
                               max_depth=8, max_features='sqrt', random_state=123, n_jobs=1)
    kf = KFold(n_splits=3, shuffle=True)
    error = []
    print "begin crossval"
    for train_index, test_index in kf.split(y1):
        y_train, y_test = y1[train_index], y1[test_index]
        X_train, X_test = X1.loc[train_index], X1.loc[test_index]
        print y_test
        X_train = X_train.fillna(0)
        y_train = y_train.fillna(0)
        X_test = X_test.fillna(0)
        y_test = y_test.fillna(0)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(np.arange(len(y_test)), y_test, label='data')
        ax.plot(np.arange(len(y_test)), predictions)
        plt.legend()
        plt.show()
        error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
    print "updating grid"
    global grid
    grid.append([np.mean(error), 1, sorted(
        zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True)])


def RF_model_loop(X, y, n_estimator, boot, max_dep, maxf):
    print "setting up the model"
    rf = RandomForestRegressor(n_estimators=50, bootstrap=boot,
                               max_depth=max_dep, max_features=maxf, random_state=123, n_jobs=1)
    kf = KFold(n_splits=4, shuffle=True)
    error = []
    print "begin crossval"
    for train_index, test_index in kf.split(y1):
        print test_index
        y_train, y_test = y.values[train_index], y.values[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        # X_train = X_train.fillna(0)
        # y_train = y_train.fillna(0)
        # X_test = X_test.fillna(0)
        # y_test = y_test.fillna(0)
        print X_train, y_train
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(np.arange(len(y_test)), y_test, label='data')
        ax.plot(np.arange(len(y_test)), predictions)
        plt.legend()
        plt.show()
        error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
    print "updating grid"
    global grid
    grid.append([np.mean(error), 'max_dep', max_dep, ' boot', boot, ' mf', maxf, sorted(
        zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True)])


max_dep = [3, 4, 5, 6, 8, 10, None]
estimators = [100]
# , 550, 600, 650, 700, 750, 800, 850, 900]
bootstrap = [False, True]
max_feat = ['log2', 'sqrt']
clusters = [[X1, y1], [X2, y2], [X3, y3], [X4, y4], [X5, y5]]
# print clusters

for estimator in estimators:
    for depth in max_dep:
        for boot in bootstrap:
            for maxf in max_feat:
                cluster_id = 1
                for XC, yc in clusters:
                    grid.append(cluster_id)
                    RF_model_loop(XC, yc, estimator, boot, depth, maxf)
                    cluster_id += 1
for item in grid:
    print item
