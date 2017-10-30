import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from statsmodels.tsa.seasonal import seasonal_decompose


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/train_clean.csv',
                 parse_dates=['date'], date_parser=dateparse)


df = df.sort_values('date')
df = df[df.rounds < 200]


features = ['DOW', 'DOY', 'temp_high', 'hum_avg', 'prec']
X = df[features]
# features = ['DOW', 'DOY', 'temp_high',
# 'temp_low', 'hum_avg', 'wind_avg', 'prec'] # best features so far

for feat1 in features:
    for feat2 in features:
        if feat1 != feat2:
            X[feat1 + '_' + feat2] = X[feat1] * X[feat2]

for feat1 in features:
    for feat2 in features:
        if feat1 != feat2:
            X[feat1 + '^2_' + feat2] = X[feat1]**2 * X[feat2]

X['temp_high_sqrt'] = np.sqrt(X['temp_high'])
X['hum_avg_sqrt'] = np.sqrt(X['hum_avg'])

print X.columns
features_new = ['DOY', 'temp_high^2_DOY', 'DOY^2_temp_high', 'temp_high_sqrt', 'temp_high', 'DOY^2_hum_avg', 'DOY_hum_avg', 'DOY_temp_high', 'hum_avg^2_DOY', 'temp_high^2_hum_avg',
                'temp_high_hum_avg', 'hum_avg', 'DOY^2_DOW', 'hum_avg_sqrt', 'DOY_DOW', 'DOW^2_DOY', 'hum_avg^2_DOW', 'temp_high^2_DOW', 'hum_avg_prec', 'prec^2_hum_avg', 'prec_DOW', 'DOW_temp_high']

features_new2 = ['DOW^2_temp_high', 'prec^2_DOW', 'prec_hum_avg',
                 'prec', 'hum_avg^2_prec', 'temp_high^2_prec', 'hum_avg_DOW', 'hum_avg^2_temp_high']

new = features_new + features_new2


X = X[new]
# X['prec_DOY'] = X['prec'] * X['DOY']
# X['hum_avg^2_prec'] = X['hum_avg']**2 * X['prec']
# X['hum_avg_prec'] = X['hum_avg'] * X['prec']
# X['temp_high^2_hum_avg'] = X['temp_high']**2 * X['hum_avg']
# X['temp_high_DOW'] = X['temp_high'] * X['DOW']


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
rounds_series = datetime_series(df, 'rounds')
rounds_decomposition = seasonal_decompose(
    rounds_series.values, model='additive', freq=275)
rounds_residual = rounds_decomposition.observed - rounds_decomposition.seasonal


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


weekly = smooth(rounds_residual, 7)

y = df['rounds'].values
# y = pd.Series(rounds_residual_smooth)
grid = []
y = weekly


def fit_linear_model(a):
    print "setting up the model"
    lr = Lasso(alpha=a, copy_X=True, normalize=True)
    kf = KFold(n_splits=5, shuffle=True)
    error = []
    print "begin crossval"
    for train_index, test_index in kf.split(y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(np.arange(len(y_test)), y_test, label='data')
        ax.plot(np.arange(len(y_test)), predictions)
        plt.legend()
        plt.show()
        error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
    print "updating grid"
    global grid
    grid.append([np.mean(error), a, sorted(
        zip(X.columns, lr.coef_), key=lambda x: x[1], reverse=True)])


alphas = [0.0001, 0.001, 0.005]
solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
for alpha in alphas:
    # for solver in solvers:
    fit_linear_model(alpha)

with open('/home/mike/golf-staff/models/lasso_results.txt', 'w') as f:
    for row in sorted(grid):
        f.write(str(row))
        f.write('\n\n')


def RF_model(num_estimators, max_dep, boot, max_feat):
    print "setting up the model"
    rf = RandomForestRegressor(n_estimators=num_estimators, bootstrap=boot,
                               max_depth=max_dep, max_features=max_feat, random_state=123, n_jobs=4)
    kf = KFold(n_splits=5, shuffle=True)
    error = []
    print "begin crossval"
    for train_index, test_index in kf.split(y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
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
    grid.append([np.mean(error), num_estimators, max_dep, ' boot', boot, ' mf', max_feat, sorted(
        zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)])


grid = []
max_dep = [None]
estimators = [500]
bootstrap = [False, 'True']
max_feat = ['log2', 'sqrt']
for estimator in estimators:
    for depth in max_dep:
        for boot in bootstrap:
            for maxf in max_feat:
                RF_model(estimator, depth, boot, maxf)

with open('/home/mike/golf-staff/models/weekly_random_forest_results.txt', 'w') as f:
    for row in sorted(grid):
        f.write(str(row))
        f.write('\n\n')

print "end"
