import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


data = pd.read_csv('data_clean.csv', parse_dates=[
                   'date'], date_parser=dateparse)
y = np.zeros(data.shape[0])
len(y)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1)

X_train.head()
# X_train.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1).to_csv(
#     'train_nznt.csv', index=False)
# X_test.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1).to_csv(
#     'test_nznt.csv', index=False)

X_train.to_csv('train_clean.csv', index=False)
X_test.to_csv('test_clean.csv', index=False)
