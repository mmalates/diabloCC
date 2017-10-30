import pandas as pd
import numpy as np

df = pd.read_csv('weather.csv')
df = df.drop_duplicates()
df['prec'] = df['prec'].fillna(0)
rain = df['prec'].values
maskT = rain == 'T'
rain[maskT] = 0.001
cum_rain = np.zeros(len(rain))
rain = rain.astype(float)
cum_rain[0] = rain[0]
cum_rain[1] = rain[0]
for i in np.arange(2, len(cum_rain), 1):
    cum_rain[i] = rain[i - 1] + rain[i - 2]
cum_rain
print np.sum(cum_rain)

df['cum_rain'] = cum_rain


df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1).to_csv(
    'weather_cum.csv', index=False)
