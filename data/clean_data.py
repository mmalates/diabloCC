import pandas as pd
import numpy as np


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('data_cum.csv', parse_dates=[
    'date'], date_parser=dateparse)
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

# sqrt features
df['temp_high_sqrt'] = np.sqrt(df['temp_high'])
df['hum_avg_sqrt'] = np.sqrt(df['hum_avg'])
df['prec_sqrt'] = np.sqrt(df['prec'])

df['temp_high_sqrt'] = df['temp_high_sqrt'].round(2)
df['hum_avg_sqrt'] = df['hum_avg_sqrt'].round(2)
df['prec_sqrt'] = df['prec_sqrt'].round(2)

df.to_csv('data_cum_clean.csv')
