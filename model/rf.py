import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()
kf = KFold(n_splits=10)
accuracy = []
recall = []
precision = []

for train_index, test_index in kf.split(y):
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test = X[train_index], X[test_index]
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy.append(skm.accuracy_score(y_test, predictions))
    recall.append(skm.recall_score(y_test, predictions))
    precision.append(skm.precision_score(y_test, predictions))

print np.mean(accuracy), np.mean(recall), np.mean(precision)
