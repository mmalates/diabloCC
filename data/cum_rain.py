import pandas as pd
import numpy as np

df = pd.read_csv('diablo_nznt.csv')

rain = df['prec'].values

cum_rain = np.zeros(len(rain))

cum_rain[0] = rain[0]
cum_rain[1] = rain[0]
for i in np.arange(2, len(cum_rain), 1):
    cum_rain[i] = rain[i - 1] + rain[i - 2]

print sum(cum_rain)

df['cum_rain'] = cum_rain
df.head()

df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1).to_csv(
    'diablo_nznt_cum.csv', index=False)
