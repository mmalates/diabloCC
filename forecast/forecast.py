import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor


def model():
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
    df240['fcst_valid_local'] = pd.to_datetime(df240['fcst_valid_local'])
    df240['date'] = df240.fcst_valid_local.dt.date
    averaged = df240.groupby(by='date').mean()
    averaged.head()
    averaged.columns
    df240 = averaged[['vis', 'dewpt', 'mslp']]
    df240['date'] = df240.index
    df240['date'] = pd.to_datetime(df240['date'])
    df240['date'] = df240.date.dt.strftime('%m-%d %a')
    print df240
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
            # dow = pd.to_datetime(date,'%Y-%m-%d')
            tenday.append([date, temp_high, temp_low, humid, wind, prec])

    data = pd.DataFrame(tenday, columns=[
                        'date', 'temp_high', 'temp_low', 'hum_avg', 'wind_avg', 'prec'])

    data['date'] = pd.to_datetime(data['date'])
    data['DOW'] = data.date.dt.dayofweek
    data['DOY'] = data.date.dt.dayofyear
    data['date'] = data.date.dt.strftime('%m-%d %a')
    data = data.merge(df240, how='left', on='date')
    data['vis_avg'] = data['vis']
    data['dew_avg'] = data['dewpt']
    data['sea_avg'] = data['mslp']

    def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv('../data/train_clean.csv',
                     parse_dates=['date'], date_parser=dateparse)

    features = ['DOW', 'DOY', 'temp_high', 'temp_low',
                'hum_avg', 'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg']

    X = df[features]
    y = df['rounds']
    data['prediction'] = RF_model(X, y, data).astype(int)
    return data
    # return render_template('model.html', title='Ten Day Forecast', tenday_fc=data.values, user=user_info, forecast_data={}, user_photo=user_photo_url, prediction=data.prediction.values)


def RF_model(X, y, forecast):
    rf = RandomForestRegressor(n_estimators=50, bootstrap=False,
                               max_depth=8, max_features='log2', random_state=123, n_jobs=4)
    rf.fit(X, y)
    return rf.predict(forecast[['DOW', 'DOY', 'temp_high', 'temp_low', 'hum_avg', 'wind_avg', 'prec', 'sea_avg', 'dew_avg', 'vis_avg']])


print model()
