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
X = dataframe.iloc[:, 0:-1]
X = X.to_numpy()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = dataframe['diagnosis']
y = y.to_numpy()
# print(X.shape)
# print(y.shape)

print(X)

hidden_layer_sizes = [(50,), (100,), (50,100), (100, 100), (25,), (25,50)]
scores = {}

for l in hidden_layer_sizes:
  mlp = MLPClassifier(hidden_layer_sizes=l, random_state=1410, max_iter=400)
  score = cross_val_score(mlp, X, y, cv=10).mean()
  scores[l] = score
  print("warstwy %s dokladnosc: %2.2f" %(str(l), 100*score))

plt.figure(figsize=(20,10))
plt.ylim([0.7,0.9])
for k,v in scores.items():
  plt.bar(str(k),v)
plt.show()

solvers = ['sgd', 'adam', 'lbfgs']
scores = {}

for s in solvers:
  mlp = MLPClassifier(hidden_layer_sizes=(50,), solver=s, random_state=1410, max_iter=1000)
  score = cross_val_score(mlp, X, y, cv=10).mean()
  scores[s] = score
  print("alogrytm %s dokladnosc: %2.2f" %(str(l), 100*score))

  plt.figure(figsize=(20,10))
plt.ylim([0.5,0.9])
for k,v in scores.items():
  plt.bar(str(k),v)
plt.show()