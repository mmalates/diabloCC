import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from numpy.linalg import LinAlgError

%matplotlib inline
rcParams['figure.figsize'] = 15, 6
pd.options.display.max_rows = None


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


def load_datetime_series(filename, target):
    # load data
    data = pd.read_csv(filename, parse_dates=[
                       'year_month'], date_parser=dateparse)

    data = data.drop_duplicates()
    data['year_month'] = data.year_month.dt.strftime('%Y-%m')

    data = data.groupby(by='year_month').sum()

    # make month_year the index for datetime series creation
    data.index = pd.to_datetime(data.index.astype(str), format='%Y-%m')

    # sort by date
    data = data.sort_index()

    # get datetime series
    return data[target]


def fill_in_missing_months(series):
    mask = [False if x.split(
        '-')[1] != '07' else True for x in series.index.astype(str)]
    series[pd.Timestamp('2012-07-01')] = np.mean(series[mask])
    series = series.sort_index()
    mask = [False if x.split(
        '-')[1] != '01' else True for x in series.index.astype(str)]
    series[pd.Timestamp('2015-01-01')] = np.mean(series[mask])
    series = series.sort_index()
    return series

# create a differenced series


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# invert differenced value


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# correct for rolling mean


def subtract_rolling_mean(series):
    moving_avg = series.rolling(window=12, center=True).mean()
    series = series - moving_avg
    series.dropna(inplace=True)
    return series


def _safe_arma_fit(y, order, trend, start_params=None):
    try:
        return ARMA(y, order=order).fit(disp=0, method='mle', trend=trend,
                                        start_params=start_params)
    except LinAlgError:
        # SVD convergence failure on badly misspecified models
        print 'LinAlgError'
        return

    except ValueError as error:
        if start_params is not None:  # don't recurse again
            # user supplied start_params only get one chance
            return
        # try a little harder, should be handled in fit really
        elif ('initial' not in error.args[0] or 'initial' in str(error)):
            start_params = [.1] * sum(order)
            if trend == 'c':
                start_params = [.1] + start_params
            return _safe_arma_fit(y, order, trend,
                                  start_params)
        else:
            return
    except:  # no idea what happened
        return


if __name__ == '__main__':
    output = []
    filepath = '../../data/guest.csv'
    targets = ['guest_rounds', 'family_guest', 'club_cars']
    for target in targets:
        series = load_datetime_series(filepath, target)
        series = fill_in_missing_months(series)
        # series = subtract_rolling_mean(series)
        X = series.values
        X = X.astype('float32')
        # validation
        errors = []
        yearly_errors = []
        for back_to_the_future in range(-4, -2):
            method = 'css-mle'
            start_params = None
            start_ar_lags = None
            order = (12, 1, 1)
            months_in_year = 12
            history = [x for x in series[months_in_year * back_to_the_future -
                                         12:len(series) - months_in_year * back_to_the_future + 12]]
            predictions = list()
            forecast_months = 12
            test = series[months_in_year * back_to_the_future +
                          12: months_in_year * back_to_the_future + 24]
            output.append("-".join(str(test.index[0]).split('-')[0:2]))
            for i in range(12):
                diff = difference(history, months_in_year)
                # predict
                model_fit = _safe_arma_fit(
                    diff, order, 'nc', start_params=None)
                # model = ARIMA(diff, order=order)
                # model_fit = model.fit(
                # trend='nc', method=method, start_params=start_params,  start_ar_lags=start_ar_lags, disp=0)
                if model_fit == None:
                    yhat = 0
                else:
                    yhat = model_fit.forecast()[0]
                yhat = inverse_difference(history, yhat, months_in_year)
                predictions.append(yhat)
                # observation
                obs = test[i]
                history.append(obs)
                output.append('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
            # report performance
            mse = mean_squared_error(test, predictions)
            rmse = np.sqrt(mse)
            total = sum(predictions)
            output.append(('total rounds: {}').format(total))
            output.append(('actual rounds: {}').format(sum(test)))
            output.append(
                ('Yearly error: {}'.format(np.abs(sum(test) - total))))
            year_diff = np.abs(sum(test) - total)
            output.append('RMSE: %.3f' % rmse)
            errors.append(rmse)
            yearly_errors.append(year_diff)
        output.append("CV RMSE: {}".format(np.mean(errors)))
        output.append("Average Yearly Error: {}".format(
            np.mean(yearly_errors)))
        # forecast
        history = [x for x in series[-24:]]
        predictions = list()
        forecast_months = 12
        prev_year = series[-12:]
        for i in range(forecast_months):
            # difference data
            months_in_year = 12
            diff = difference(history, months_in_year)
            # predict
            model_fit = _safe_arma_fit(diff, order, 'nc', start_params=None)
            # model = ARIMA(diff, order=order)
            # model_fit = model.fit(
            #     trend='nc', method=method, start_params=start_params,   start_ar_lags=start_ar_lags, disp=0)

            if model_fit == None:
                yhat = 0
            else:
                yhat = model_fit.forecast()[0]
            yhat = inverse_difference(history, yhat, months_in_year)
            predictions.append(yhat)
            obs = prev_year[i]
            history.append(obs)
            output.append('>Predicted=%.3f' % yhat)
        total = sum(predictions)
        output.append(('forecasted ' + target +
                       ' rounds: {}').format(total))
        output.append('----------------------------------------')
    with open('/home/mike/diabloCC/models/arima/guest_results.txt', 'w') as f:
        for row in output:
            f.write(str(row))
            f.write('\n')


# BASELINE persistence
    output_base = []
    for target in targets:
        series = load_datetime_series(filepath, target)
        series = fill_in_missing_months(series)
        # series = subtract_rolling_mean(series)
        X = series.values
        X = X.astype('float32')
        # validation
        errors = []
        yearly_errors = []
        for back_to_the_future in range(-5, -2):
            months_in_year = 12
            # print series[months_in_year * back_to_the_future - 12]
            history = [x for x in series[months_in_year * back_to_the_future: len(
                series) - months_in_year * back_to_the_future + months_in_year]]
            predictions = list()
            forecast_months = 12
            test = series[months_in_year * back_to_the_future +
                          months_in_year: months_in_year * back_to_the_future + 2 * months_in_year]
            output_base.append("-".join(str(test.index[0]).split('-')[0:2]))
            # diff = difference(history, months_in_year)
            # # yhat = []
            # for i in range(12):
            #     # predict
            #     # model = ARIMA(diff, order=(0, 1, 1))
            #     # model_fit = model.fit(trend='nc', disp=0)
            #     # yhat = model_fit.forecast()[0]
            #     yhat.append(diff[i])
            #     obs = test[i]
            #     history.append(obs)
            # yhat = inverse_difference(history, yhat, months_in_year)
            for pred, exp in zip(history, test):
                predictions.append(pred)
                # observation
                output_base.append(
                    '>Predicted=%.3f, Expected=%3.f' % (pred, exp))
            # report performance
            mse = mean_squared_error(test, predictions)
            rmse = np.sqrt(mse)
            total = sum(predictions)
            output_base.append(('total rounds: {}').format(total))
            output_base.append(('actual rounds: {}').format(sum(test)))
            output_base.append(
                ('Yearly error: {}'.format(np.abs(sum(test) - total))))
            year_diff = np.abs(sum(test) - total)
            output_base.append('RMSE: %.3f' % rmse)
            errors.append(rmse)
            yearly_errors.append(year_diff)
        output_base.append("CV RMSE: {}".format(np.mean(errors)))
        output_base.append(
            "Average Yearly Error: {}".format(np.mean(yearly_errors)))
        # forecast
        history = [x for x in series[-12:]]
        predictions = list()
        forecast_months = 12
        # prev_year = series[-12:]
        for i in range(forecast_months):
            # # difference data
            # months_in_year = 12
            # diff = difference(history, months_in_year)
            # # predict
            # model = ARIMA(diff, order=(1, 1, 0))
            # model_fit = model.fit(trend='nc', disp=0)
            # yhat = diff[i]
            # yhat = inverse_difference(history, yhat, months_in_year)
            predictions.append(history[i])
            # obs = prev_year[i]
            # history.append(obs)
            output_base.append('>Predicted=%.3f' % history[i])
        total = sum(predictions)
        output_base.append(
            ('forecasted ' + target + ' rounds: {}').format(total))
        output_base.append('----------------------------------------')
    with open('/home/mike/diabloCC/models/arima/baseline_results.txt', 'w') as f:
        for row in output_base:
            f.write(str(row))
            f.write('\n')
