import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import json
from collections import defaultdict

url = 'https://api.weather.com/v2/turbo/vt1dailyforecast?apiKey=d522aa97197fd864d36b418f39ebb323&format=json&geocode=43.59%2C-116.62&language=en-US&units=e'

response = requests.get(url)
jsoned = response.json()
for item, value in jsoned.iteritems():
    if item == 'vt1dailyforecast':
        # print value
        for key, value2 in value.iteritems():
            print key, value2
forecast = pd.DataFrame()
for item, value in jsoned.iteritems():
    if item == 'vt1dailyforecast':
        # print value
        for key, value2 in value.iteritems():
            # print key, value2
            if key == 'validDate':
                # print value2
                forecast[key] = np.array(value2)
            if key == 'day':
                for feature, val in value2.iteritems():
                    # print feature, val
                    forecast[feature] = np.array(val)

forecast.head()
forecast.columns
features = [u'validDate', u'humidityPct', u'temperature', u'cloudPct',
            u'windSpeed', u'precipPct', u'precipAmt', u'dayPartName', u'uvIndex']

df = forecast[forecast['dayPartName'] != 'Today'][features]

df.head()
