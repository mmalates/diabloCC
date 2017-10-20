import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from statsmodels.tsa.seasonal import seasonal_decompose


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('~/golf-staff/data/train_clean.csv',
                 parse_dates=['date'], date_parser=dateparse)


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


seasonal_smooth = smooth(rounds_decomposition.seasonal, 30)

rounds_residual_smooth = rounds_decomposition.observed - seasonal_smooth


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
y = df['rounds'].values
# y = pd.Series(rounds_residual_smooth)
grid = []


def CAT_model():
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.02)
# approx_on_full_history=False, gradient_iterations=1,
    # loss_function='RMSE', iterations=1000, learning_rate=0.03, thread_count=None, depth=8, border_count=255)
    # cat = CatBoostRegressor(iterations=estimators,
    # depth=depth,
    # learning_rate=learn_rate, loss_function='RMSE')
    kf = KFold(n_splits=10, shuffle=False)
    error = []
    print "starting stuff"
    for train_index, test_index in kf.split(y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        cat.fit(X_train, y_train)
        print "made fit"
        predictions = cat.predict(X_test)
        print "made prediction"
        error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
    print "got stuff"
    global grid
    grid.append([np.mean(error), 'Cat', sorted(
        zip(X.columns, cat.feature_importances_), key=lambda x: x[1], reverse=True)])
    print "made it"


CAT_model()

with open('grid_results/CAT_results.txt', 'w') as f:
    for row in sorted(grid):
        f.write(str(row))
        f.write('\n\n')
