import pandas as pd


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


golf = pd.read_csv('diablo.csv', parse_dates=['date'], date_parser=dateparse)

weather = pd.read_csv('weather.csv', parse_dates=[
                      'date'], date_parser=dateparse)

golf.head()
weather.head()

data = pd.merge(golf, weather, on='date', suffixes=(
    '_golf', '_weather'), how='left')

data.columns
data = data.drop(['Unnamed: 0_golf', 'Unnamed: 0.1_golf',
                  'Unnamed: 0_weather', 'Unnamed: 0.1_weather'], axis=1)

data.to_csv('data.csv')
data.head()
