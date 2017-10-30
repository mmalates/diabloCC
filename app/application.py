from flask import Flask
from flask import render_template
import requests
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import pickle


application = Flask(__name__)


@application.route('/')
def landing():
    return render_template("base.html")


brayden_features = ['temp_high', 'temp_low',
                    'hum_avg', 'wind_avg', 'vis_avg']

# define features for clustering weather data
cluster_features = ['temp_high', 'temp_low',
                    'hum_avg', 'wind_avg', 'prec', 'vis_avg']

# define features for modelling
model_features = ['year', 'month', 'DOW', 'DOY', 'temp_high', 'temp_low',
                  'hum_avg', 'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg', 'temp_high_sqrt', 'rain', 'cum_prec', 'hum_avg_sqrt', 'prec_sqrt']

perfect_weather = np.array([71.1, 46.3, 58.5, 6.3, 10])


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
    data['year'] = data.date.dt.year
    data['month'] = data.date.dt.month
    data['date_for_table'] = data.date.dt.strftime('%m-%d %a')

    # create rain boolean
    data['rain'] = data.prec.values > 0

    # create cumulative rain feature
    prec_arr = data.prec.values
    prec_arr = np.insert(prec_arr, 0, 0.)[:-1]
    data['cum_prec'] = np.array(prec_arr) + data.prec.values

    # remove mondays
    not_mondays = data.DOW != 0
    data = data[not_mondays]

    global model_features
    global perfect_weather
    p_weather_diff = []

    for day in data[brayden_features].values:
        p_weather_diff.append(get_p_weather_diff(day, perfect_weather))

    data['p_weather_diff'] = np.array(p_weather_diff)
    data['p_weather_diff'] = 10**(-0.2 * data['p_weather_diff'])
    with open('data/rf.pkl', 'rb') as pkl_file:
        rf = pickle.load(pkl_file)
    predictions = rf.predict(data[model_features])
    # create predictions columns
    data['prediction'] = predictions.astype(int)
    data['crowd_term'] = (1.0) * (1.0 / (data['prediction']**(1. / 2.)))
    golfability = np.zeros(len(data.prediction))
    for i, crowd in enumerate(data.prediction.values):
        if crowd >= 30:
            golfability[i] = 80 * data['crowd_term'].values[i] * \
                data['p_weather_diff'].values[i] / \
                (1 + data['prec'].values[i])**1.4
        else:
            golfability[i] = 11 * data['p_weather_diff'].values[i] / \
                (1 + data['prec'].values[i])**1.4
    data['golfability'] = golfability
    data['golfability'] = data['golfability'].round(2)
    data['p_weather_diff'] = data['p_weather_diff'].round(2)
    data['crowd_term'] = data['crowd_term'].round(2)
    return render_template('model.html', tenday_fc=data.values)


def get_p_weather_diff(day, perfect_weather):
    diff = np.sum(np.abs(np.subtract(day, perfect_weather) / perfect_weather))
    return diff


if __name__ == "__main__":
    application.run()
