import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.model_selection import KFold


df = pd.read_csv('../data/train_nznt.csv')
df.head()

kf = KFold(n_splits=10, shuffle=True)
error = []
y = df['rounds']

for train_index, test_index in kf.split(y):
    y_train, y_test = y[train_index], y[test_index]
    predictions = np.ones(len(y_test)) * np.mean(y_train)
    error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))

print np.mean(error)
