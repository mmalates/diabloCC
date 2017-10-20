import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestRegressor


# read training data
def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/train_clean.csv',
                 parse_dates=['date'], date_parser=dateparse)

# feature engineering
df = df.sort_values('date')
df['year'] = df.date.dt.year
df['month'] = df.date.dt.month

# create rain boolean
df['rain'] = df.prec.values > 0

# create cumulative rain feature
prec_arr = df.prec.values
prec_arr = ([0.0] + list(prec_arr))[:-1]
df['cum_prec'] = np.array(prec_arr) + df.prec.values

# create rain the day before boolean
rain_arr = df.rain.values
rain_dbf = ([False] + list(rain_arr))[:-1]
df['rain_db4'] = np.array(rain_dbf)

# remove outliers
df = df[df.rounds < 200]

# remove mondays because they are all holidays
not_mondays = df.DOW != 0
df = df[not_mondays]

# create feature list for modeling
features = ['year', 'month', 'rain', 'DOW', 'DOY', 'temp_high', 'temp_low', 'hum_avg',
            'wind_high', 'prec', 'sea_high', 'dew_avg', 'vis_low', 'rain_db4', 'cum_prec']

# select features from training data
X = df[features]

# use to investigate feature interactions
# for feat1 in features:
#     for feat2 in features:
#         if feat1 != feat2:
#             X[feat1 + '_' + feat2] = X[feat1] * X[feat2]
#
# for feat1 in features:
#     for feat2 in features:
#         if feat1 != feat2:
#             X[feat1 + '^2_' + feat2] = X[feat1]**2 * X[feat2]

# more features
X['temp_high_sqrt'] = np.sqrt(X['temp_high'])
X['hum_avg_sqrt'] = np.sqrt(X['hum_avg'])
X['prec_sqrt'] = np.sqrt(X['prec'])


# target variable
y = df['rounds']


def RF_model(num_estimators, max_dep, boot, max_feat, max_leaf):
    '''
    Grid search random forest regressor.

    Input params: number of estimators, maximum depth, bootstrap, maximum features, minimum leaf sample size.

    Output: None, Populates "grid" global list variable
    '''
    print "setting up the model"
    rf = RandomForestRegressor(n_estimators=num_estimators, bootstrap=boot, criterion='mae', oob_score=True,
                               min_samples_leaf=max_leaf,
                               max_depth=max_dep, max_features=max_feat, random_state=123, n_jobs=4)
    kf = KFold(n_splits=10, shuffle=True)
    error = []
    oob = []
    print "begin crossval"
    for train_index, test_index in kf.split(y):
        y_train, y_test = y.values[train_index], y.values[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        print np.sqrt(skm.mean_squared_error(y_test, predictions))
        error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
        oob.append(rf.oob_score_)
    print "updating grid"
    global grid
    grid.append([np.mean(error), 'oob', np.mean(oob), 'leaf_size', max_leaf, num_estimators, max_dep, ' boot', boot, ' mf', max_feat, sorted(
        zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)])


# define global grid list variable
grid = []

# define grid search params
max_dep = [8]
estimators = [100]
bootstrap = [True]
max_feat = [0.6]
max_leaf = [8]

for estimator in estimators:
    for depth in max_dep:
        for boot in bootstrap:
            for maxf in max_feat:
                for leaf_size in max_leaf:
                    RF_model(estimator, depth, boot, maxf, leaf_size)


with open('/home/mike/golf-staff/models/10-19_feature_results_score.txt', 'w') as f:
    for row in sorted(grid):
        f.write(str(row))
        f.write('\n\n')

print "end"
