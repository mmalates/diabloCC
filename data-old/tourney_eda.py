import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


df = pd.read_csv('data.csv', parse_dates=[
    'date'], date_parser=dateparse)
df.head()
df.describe().T
df.tourney.describe()
df.rounds.describe()

fig, ax = plt.subplots()
ax.plot(data.tourney.values)
plt.show()

mask_neg_round = df.rounds < 0
mask_neg_tourn = df.tourney < 0
df.tourney.loc[mask_neg_tourn] = 0
df.rounds.loc[mask_neg_round] = 0
df.tourney.describe()
df.rounds.describe()

days = df.date.values
rounds = df.rounds.values
no_zero = rounds > 0

days_no_zero = days[no_zero]
rounds_no_zero = rounds[no_zero]

walkers = df.walkers.values
riders = rounds - walkers
tourneys = df.tourney.values
round_minus_tourney = rounds - tourneys
no_zero_or_tourney = round_minus_tourney > 0
df_no_zero_or_tourney = df[no_zero_or_tourney]

df_no_zero_or_tourney.describe().T
mask_tourney = df_no_zero_or_tourney.tourney == 0
df_clean = df_no_zero_or_tourney[mask_tourney]
df_clean.loc[df_clean.prec == 'T', 'prec'] = 0.001
df_clean.prec = df_clean.prec.astype(float)
df_clean.describe().T
df_clean['DOY'] = df_clean.date.dt.dayofyear
df_clean = df_clean.drop('tourney', axis=1)
df_clean = df_clean.fillna(0)
df_clean = df_clean.drop('Unnamed: 0', axis=1)

df_clean.to_csv('data_clean.csv', index=False)
