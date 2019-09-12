import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import tensorflow.python.framework.dtypes

a = tf.constant(2)
b = tf.constant(3)
d = a*b

data = pd.read_csv("eurusd2.csv", sep=',',index_col=None, names=["date", "open", "close", "min", "max", "max2", "sec1", "sec2", "sec3", "sec4", 'sec5'])
data.drop('max2', axis=1, inplace=True)
data.drop('sec5', axis=1, inplace=True)

print(d)
print(data.head())

X = data.copy()
X.drop('open', axis=1, inplace=True)

y=data['open']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 50)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

Check_test = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred.flatten()})
r2=r2_score(y_test, y_pred)

print(X_test)

