import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data.csv")
dataframe.pop('id')
dataframe['diagnosis'] = dataframe['diagnosis'].replace({'M': '0'})
dataframe['diagnosis'] = dataframe['diagnosis'].replace({'B': '1'})
print(dataframe)
X = dataframe.iloc[:, 0:-1]
X = X.to_numpy()

# y = dataframe['DEATH_EVENT']
# y = y.to_numpy()
print(X.shape)
# print(y.shape)