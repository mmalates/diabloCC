import pandas as pd
import numpy as np


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


# load golf, weathe, and notes data
data = pd.read_csv('data_clean.csv', parse_dates=[
                   'date'], date_parser=dateparse)
weather = pd.read_csv('weather_cum.csv', parse_dates=[
    'date'], date_parser=dateparse)
notes = pd.read_csv('notes.csv', parse_dates=[
    'Day'], date_parser=dateparse)

# merge tables
notes['date'] = notes['Day']
notes = notes.fillna(" ")
notes = notes.drop_duplicates()
notes = notes.drop('Day', axis=1)
cum_raindf = pd.DataFrame(weather[['date', 'cum_rain']])
cum_rain_notes = notes.merge(cum_raindf, how='left', on='date')
cum_rain_notes = cum_rain_notes.drop_duplicates()
merged = data.merge(cum_rain_notes, how='right', on='date')
merged = merged.dropna()
merged = merged.drop_duplicates()

# save to files
data.to_csv('data_cum.csv', index=False)
merged.to_csv('data_notes.csv', index=False)
