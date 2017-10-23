import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


from statsmodels.tsa.seasonal import seasonal_decompose


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('../data/train_clean.csv',
                 parse_dates=['date'], date_parser=dateparse)

df = df.sort_values('date')

features = [['DOW', 'DOY', 'temp_high', 'temp_low',
             'hum_avg', 'wind_avg', 'prec'], ['DOW', 'DOY', 'temp_high', 'temp_low', 'prec'], ['DOW', 'DOY', 'temp_high', 'temp_low',
                                                                                               'hum_avg', 'wind_avg'], ['DOW', 'DOY', 'temp_high', 'temp_low']]


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


#y = df['rounds']
# y = pd.Series(rounds_residual_smooth)
y = pd.Series(rounds_residual)
grid = []
inter = [True, False]


def build_linear_model(inter):
    lm = LinearRegression(normalize=True, fit_intercept=inter)
    kf = KFold(n_splits=10, shuffle=False)
    error = []

    for train_index, test_index in kf.split(y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)
        error.append(np.sqrt(mean_squared_error(y_test, predictions)))

    global grid
    grid.append([np.mean(error), sorted(
        zip(feat, lm.coef_), key=lambda x: x[1], reverse=True)])


for int in inter:
    for feat in features:
        X = df[feat]
        build_linear_model(int)

with open('/home/mike/golf-staff/models/grid_linear.txt', 'w') as f:
    for row in sorted(grid):
        f.write(str(row))
        f.write('\n\n')

df.head()
print "end"
