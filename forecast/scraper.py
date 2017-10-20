import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import json
from collections import defaultdict

url240 = 'https://api.weather.com/v1/geocode/37.82616/-121.980217/forecast/hourly/240hour.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'


response240 = requests.get(url240)

print type(response240.content)
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
df240.head()


url = 'https://api.weather.com/v1/geocode/37.821373/-121.968079/forecast/daily/10day.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'

response = requests.get(url)

print type(response.content)
jsoned = response.json()
tenday = []
for item in jsoned['forecasts']:
    for key, value in item.iteritems():
        # print key,':', value
        if key == 'qpf':
            print key, '-', value
            prec = value
        if key == 'min_temp':
            print key, '-', value
            temp_low = value
        if key == 'fcst_valid_local':
            print key, '-', value.split('T')[0]
            date = value.split('T')[0]
        if key == 'day':
            for keyday, valday in value.iteritems():
                if keyday == 'wspd':
                    print keyday, '-', valday
                    wind = valday
                # if keyday == 'temp_phrase':
                    # print keyday, '-', valday.split(" ")[1].replace("F", "")
                    # temp_high = valday
                # if keyday == 'hi':
                #     print keyday, '-', valday
                if keyday == 'temp':
                    print keyday, '-', valday
                    temp_high = valday
                if keyday == 'rh':
                    print keyday, '-', valday
                    humid = valday

    lst = [date, temp_high, temp_low, humid, wind, prec]
    tenday.append(lst)
    print '----------------------------------------'
print outlst


def make_dictionaries(resp):
    counter = 1
    day = 1
    output = []
    for lst in resp.content.split('{'):
        if counter % 2 != 0:
            date = defaultdict(str)
        if counter >= 4:
            if counter % 2 == 0:
                date[day] = '{' + lst.replace("'}", "")
            if counter % 2 != 0:
                date[day] = lst + lst.replace("'}", "")
                output.append(date)
                day += 1
        counter += 1
    return output


dictionary = make_dictionaries(response)
print dictionary[0]
for item in dictionary:
    print item
for item in make_dictionaries(response):
    print item

data = response.json()
print data
for item in data.items():
    print item

url2 = 'https://api.weather.com/v1/geocode/37.66991/-121.759109/forecast/daily/10day.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'

response2 = requests.get(url2)

data2 = response2.json()
print data2
for item in data2.items():
    print item


data
soup = bs(response.content, 'html5lib')

soup.find('div')['class']

body = soup.find('body')

fgh = body.find('city-tenday')
print fgh
print app.contents
soup2 = bs(app.contents, 'html5lib')

for line in app.contents:
    if 'forecast-graph' in line:
        print 'got it'
