from flask import Flask
from flask import render_template, flash, redirect, url_for, session, jsonify
from flask_oauth import OAuth
import requests
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict


application = Flask(__name__)

user_info = "staff"
user_photo_url = 'https://demo.keypasco.com/res-1.2.2/img/User_ring.png'


@application.route('/')
@application.route('/landing')
def landing():
    global user_info, user_photo_url
    if user_info is not "":
        return render_template("landing.html",
                               title='Landing',
                               user=user_info,
                               user_photo=user_photo_url,
                               forecast_data={},
                               tenday_fc=[],
                               prediction=[])
    else:
        return render_template("landing.html",
                               title='Landing',
                               user_photo='https://demo.keypasco.com/res-1.2.2/img/User_ring.png',
                               forecast_data={})


@application.route('/model/', methods=['POST'])
def model():
    # get some weather features
    url240 = 'https://api.weather.com/v1/geocode/37.82616/-121.980217/forecast/hourly/240hour.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'

    response240 = requests.get(url240)
    jsoned240 = response240.json()
    forecast240 = jsoned240['forecasts']
    keys = []
    keys = forecast240[0].keys()
    matrix = []
    for item in forecast240:
        list = []
        for key, value in item.iteritems():
            list.append(value)
        matrix.append(list)
    df240 = pd.DataFrame(matrix, columns=keys)
    df240['fcst_valid_local'] = pd.to_datetime(
        df240['fcst_valid_local'])
    df240['date'] = df240['fcst_valid_local'].dt.date
    averaged = df240.groupby(by='date').mean()

    df_fcst = averaged[['vis', 'dewpt', 'mslp']]
    df_fcst['date'] = pd.to_datetime(df_fcst.index)
    df_fcst['date_for_table'] = df_fcst['date'].dt.strftime('%m-%d %a')
    # more weather features
    url = 'https://api.weather.com/v1/geocode/37.821373/-121.968079/forecast/daily/10day.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'
    response = requests.get(url)
    jsoned = response.json()
    tenday = []
    for item in jsoned['forecasts']:
        if 'day' in item.keys():
            for key, value in item.iteritems():
                if key == 'qpf':
                    prec = value
                if key == 'min_temp':
                    temp_low = value
                if key == 'fcst_valid_local':
                    date = value.split('T')[0]
                if key == 'day':
                    for keyday, valday in value.iteritems():
                        if keyday == 'wspd':
                            wind = valday
                        if keyday == 'temp':
                            temp_high = valday
                        if keyday == 'rh':
                            humid = valday
            tenday.append([date, temp_high, temp_low, humid, wind, prec])

    # create dataframe for weather data
    data = pd.DataFrame(tenday, columns=[
                        'date', 'temp_high', 'temp_low', 'hum_avg', 'wind_avg', 'prec'])

    # fix first weather df for merge
    data['date'] = pd.to_datetime(data['date'])
    data['DOW'] = data.date.dt.dayofweek
    data['DOY'] = data.date.dt.dayofyear

    # merge the weather data
    data = data.merge(df_fcst, how='left', on='date')

    # feature engineering
    data['vis_avg'] = data['vis'].round(1)
    data['dew_avg'] = data['dewpt'].round(1)
    data['sea_avg'] = data['mslp'].round(1)
    data['temp_high_sqrt'] = np.sqrt(data.temp_high.values)
    data['hum_avg_sqrt'] = np.sqrt(data['hum_avg'])
    data['prec_sqrt'] = np.sqrt(data['prec'])
    # data['date'] = pd.to_datetime(data['date'])
    data['year'] = data.date.dt.year
    data['month'] = data.date.dt.month
    data['date_for_table'] = data.date.dt.strftime('%m-%d %a')

    # create rain boolean
    data['rain'] = data.prec.values > 0

    # create cumulative rain feature
    prec_arr = data.prec.values
    prec_arr = np.insert(prec_arr, 0, 0.)[:-1]
    data['cum_prec'] = np.array(prec_arr) + data.prec.values

    # create rain the day before boolean
    rain_arr = data.rain.values
    rain_dbf = np.insert(rain_arr, 0, False)[:-1]
    data['rain_db4'] = np.array(rain_dbf)

    # remove mondays
    not_mondays = data.DOW != 0
    data = data[not_mondays]
    # load training data
    features = ['year', 'month', 'DOW', 'DOY', 'temp_high', 'temp_low',
                'hum_avg', 'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg', 'temp_high_sqrt', 'rain', 'cum_prec', 'hum_avg_sqrt', 'prec_sqrt']

    def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv('data/train_clean.csv',
                     parse_dates=['date'], date_parser=dateparse)

    # feature engineering
    df = df.sort_values('date')
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month

    # create rain boolean
    df['rain'] = df.prec.values > 0

    # create cumulative rain feature
    prec_arr = df.prec.values
    prec_arr = np.insert(prec_arr, 0, 0.)[:-1]
    df['cum_prec'] = np.array(prec_arr) + df.prec.values

    # create rain the day before boolean
    rain_arr = df.rain.values
    rain_dbf = np.insert(rain_arr, 0, False)[:-1]
    df['rain_db4'] = np.array(rain_dbf)

    # remove outliers
    df = df[df.rounds < 200]

    # remove mondays because they are all holidays
    not_mondays = df.DOW != 0
    df = df[not_mondays]

    df['temp_high_sqrt'] = np.sqrt(df['temp_high'])
    df['hum_avg_sqrt'] = np.sqrt(df['hum_avg'])
    df['prec_sqrt'] = np.sqrt(df['prec'])

    # define features for model
    features = ['year', 'month', 'DOW', 'DOY', 'temp_high', 'temp_low',
                'hum_avg', 'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg', 'temp_high_sqrt', 'rain', 'cum_prec', 'hum_avg_sqrt', 'prec_sqrt']

    # feature matrix
    X = df[features]
    # target variable
    y = df['rounds']

    # get cluster similarities
    cluster_features = ['temp_high', 'temp_low', 'hum_avg',
                        'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg']
    clusterdf = pd.read_csv('data/5ward.csv', parse_dates=[
        'date'], date_parser=dateparse)

    meanvect = clusterdf.groupby(by='label').mean()
    vectors = meanvect[cluster_features].values
    index = meanvect[cluster_features].index

    dic = defaultdict()
    for label, values in zip(index, vectors):
        dic[label] = values

    labels = []
    for day in data[cluster_features].values:
        diffs = defaultdict(float)
        for label, vector in dic.iteritems():
            diffs[label] = np.sum(np.abs(day - vector))
        labels.append(min(diffs.items(), key=lambda x: x[1]))
    label_col = []
    index_dic = {5.0: 5.0, 4.0: 2.0, 3.0: 1.0, 2.0: 4.0, 1.0: 3.0}
    for cluster_label in labels:
        label_col.append(index_dic[cluster_label[0]])
    data['labels'] = np.array(label_col)
    data['prediction'] = RF_model(X, y, data).astype(int)

    data['golf_index'] = (14.) * (1.0 / np.sqrt(data['prediction'])
                                  ) * data['labels']
    data['golf_index'] = data['golf_index'].round(2)
    # create predictions columns
    return render_template('model.html', title='Ten Day Forecast', tenday_fc=data.values, user=user_info, forecast_data={}, user_photo=user_photo_url, prediction=data.prediction.values)


def RF_model(X, y, forecast):
    ''' fit and predict with a random forest model '''
    rf = RandomForestRegressor(n_estimators=50, bootstrap=False,
                               max_depth=8, max_features=0.6, n_jobs=4)
    rf.fit(X, y)
    return rf.predict(forecast[['year', 'month', 'DOW', 'DOY', 'temp_high', 'temp_low', 'hum_avg', 'wind_avg', 'prec', 'sea_avg', 'dew_avg', 'vis_avg', 'temp_high_sqrt', 'rain', 'cum_prec', 'hum_avg_sqrt', 'prec_sqrt']])


if __name__ == "__main__":
    application.run()
