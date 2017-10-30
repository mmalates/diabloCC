import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestRegressor
import pickle


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


# load training data
df = pd.read_csv('../data/train_cum_clean.csv',
                 parse_dates=['date'], date_parser=dateparse)

# create feature list for modeling
features = ['year', 'month', 'rain', 'DOW', 'DOY', 'temp_high', 'temp_low', 'hum_avg', 'wind_high', 'prec',
            'sea_high', 'dew_avg', 'vis_low', 'cum_prec', 'temp_high_sqrt', 'hum_avg_sqrt', 'prec_sqrt']

X = df[features]
y = df['rounds']

# load test data
df_test = pd.read_csv('../data/test_cum_clean.csv',
                      parse_dates=['date'], date_parser=dateparse)

X_test_set = df_test[features]
y_test_set = df_test['rounds']


def RF_model_train_performance():
    '''
    Grid search random forest regressor.

    Input params: number of estimators, maximum depth, bootstrap, maximum features, minimum leaf sample size.

    Output: None, Prints train error and pickles model
    '''
    max_dep = 8
    estimators = 100
    bootstrap = True
    max_feat = 0.6
    max_leaf = 8
    rf = RandomForestRegressor(n_estimators=estimators, bootstrap=bootstrap, criterion='mae', oob_score=True,
                               min_samples_leaf=max_leaf,
                               max_depth=max_dep, max_features=max_feat)
    kf = KFold(n_splits=10, shuffle=True)
    error = []
    oob = []
    for train_index, test_index in kf.split(y):
        y_train, y_test = y.values[train_index], y.values[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        rf.fit(X_train, y_train)
        with open('rf.pkl', 'wb') as pkl:
            pickle.dump(rf, pkl)
        predictions = rf.predict(X_test)
        error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
        oob.append(rf.oob_score_)
    print 'train error: ', np.mean(error), 'Out of bag score: ', np.mean(oob), 'feature importances: ', sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)


def RF_model_test_performance():
    """
    Prints test set performance metric of pickled model
    Input: None
    Ouput: None, print test RMSE
    """
    global X_test_set, y_test_set
    with open('rf.pkl', 'rb') as pkl_file:
        rf = pickle.load(pkl_file)
    test_predictions = rf.predict(X_test_set)
    test_rmse = np.sqrt(skm.mean_squared_error(y_test_set, test_predictions))
    print 'test RMSE: ', test_rmse


if __name__ == '__main__':
    RF_model_train_performance()
    RF_model_test_performance()
