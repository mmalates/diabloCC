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

df = df.sort_values('date')

features = ['DOW', 'DOY', 'temp_high', 'hum_avg',
            'prec', 'wind_avg', 'vis_low', 'sea_avg', 'dew_avg']
# features = ['DOW', 'DOY', 'temp_high',
# 'temp_low', 'hum_avg', 'wind_avg', 'prec'] # best features so far


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


df = df[df.rounds < 200]

X = df[features]
for feat1 in features:
    for feat2 in features:
        if feat1 != feat2:
            X[feat1 + '_' + feat2] = X[feat1] * X[feat2]

for feat1 in features:
    for feat2 in features:
        if feat1 != feat2:
            X[feat1 + '^2_' + feat2] = X[feat1]**2 * X[feat2]

X['temp_high_sqrt'] = np.sqrt(X['temp_high'])
# X['temp_low_sqrt'] = np.sqrt(X['temp_low'])
X['hum_avg_sqrt'] = np.sqrt(X['hum_avg'])
# X['prec_DOY'] = X['prec'] * X['DOY']
# X['hum_avg^2_prec'] = X['hum_avg']**2 * X['prec']
# X['hum_avg_prec'] = X['hum_avg'] * X['prec']
# X['temp_high^2_hum_avg'] = X['temp_high']**2 * X['hum_avg']
# X['temp_high_DOW'] = X['temp_high'] * X['DOW']
# features_new = ['DOY', 'temp_high^2_DOY', 'DOY^2_temp_high', 'temp_high_sqrt', 'temp_high', 'DOY^2_hum_avg', 'DOY_hum_avg', 'DOY_temp_high', 'hum_avg^2_DOY', 'temp_high^2_hum_avg',
# 'temp_high_hum_avg', 'hum_avg', 'DOY^2_DOW', 'hum_avg_sqrt', 'DOY_DOW', 'DOW^2_DOY', 'hum_avg^2_DOW', 'temp_high^2_DOW', 'hum_avg_prec', 'prec^2_hum_avg', 'prec_DOW', 'DOW_temp_high']

# features_new2 = ['DOW^2_temp_high', 'prec^2_DOW', 'prec_hum_avg',
#  'prec', 'hum_avg^2_prec', 'temp_high^2_prec', 'hum_avg_DOW', 'hum_avg^2_temp_high']

# new = features_new + features_new2
y = df['rounds'].values
# y = pd.Series(rounds_residual_smooth)
grid = []


def CAT_model():
    cat = CatBoostRegressor()
# approx_on_full_history=False, gradient_iterations=1,
    # loss_function='RMSE', iterations=1000, learning_rate=0.03, thread_count=None, depth=8, border_count=255)
    # cat = CatBoostRegressor(iterations=estimators,
    # depth=depth,
    # learning_rate=learn_rate, loss_function='RMSE')
    kf = KFold(n_splits=8, shuffle=False)
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

with open('/home/mike/golf-staff/models/CAT_results.txt', 'w') as f:
    for row in sorted(grid):
        f.write(str(row))
        f.write('\n\n')
