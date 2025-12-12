import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# wczytanie danych

df = pd.read_csv("data.csv")

# usuniecie kolumnty id
df = df.drop(columns=["id"])

# Zakodowanie diagnozy (M=0, B=1)
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

# podzial danych

X = df.drop(columns=['diagnosis']).to_numpy()
y = df['diagnosis'].to_numpy()

# usuniecie NaN

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# skalowanie cech (0 - 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# wplyw liczby neuronow w warstwach

hidden_layer_sizes = [
    (30,), (30, 30),
    (60,), (60, 60),
    (100,),
    (120,), (120, 60)  # własne
]

scores_hidden = {}

for h in hidden_layer_sizes:
    mlp = MLPClassifier(hidden_layer_sizes=h, random_state=1410, max_iter=1500)
    score = cross_val_score(mlp, X, y, cv=10, scoring="balanced_accuracy").mean()
    scores_hidden[h] = score
    print(f"Warstwy {h} -> balanced accuracy: {score:.4f}")

# wykres
plt.figure(figsize=(16, 8))
plt.bar([str(k) for k in scores_hidden.keys()], scores_hidden.values())
plt.title("Wpływ liczby neuronów na Balanced Accuracy")
plt.ylabel("Balanced Accuracy")
plt.xticks(rotation=45)
plt.ylim(0.7, 1.0)
plt.show()

# najlepszy wariant
best_hidden = max(scores_hidden, key=scores_hidden.get)
print("Najlepsza architektura:", best_hidden)

# wplyw funkcji aktywacji

activations = ['identity', 'logistic', 'tanh', 'relu']
scores_act = {}

for a in activations:
    mlp = MLPClassifier(hidden_layer_sizes=best_hidden, activation=a,
                        random_state=1410, max_iter=1500)
    score = cross_val_score(mlp, X, y, cv=10, scoring="balanced_accuracy").mean()
    scores_act[a] = score
    print(f"Aktywacja {a} -> balanced accuracy: {score:.4f}")

# wykres
plt.figure(figsize=(16, 8))
plt.bar(scores_act.keys(), scores_act.values())
plt.title(f"Wpływ funkcji aktywacji (warstwy: {best_hidden})")
plt.ylabel("Balanced Accuracy")
plt.ylim(0.7, 1.0)
plt.show()

best_activation = max(scores_act, key=scores_act.get)
print("Najlepsza funkcja aktywacji:", best_activation)

# wplyw solvera

solvers = ['lbfgs', 'sgd', 'adam']
scores_solver = {}

for s in solvers:
    mlp = MLPClassifier(hidden_layer_sizes=best_hidden,
                        activation=best_activation,
                        solver=s,
                        random_state=1410,
                        max_iter=2000)
    score = cross_val_score(mlp, X, y, cv=10, scoring="balanced_accuracy").mean()
    scores_solver[s] = score
    print(f"Solver {s} -> balanced accuracy: {score:.4f}")

# wykres
plt.figure(figsize=(16, 8))
plt.bar(scores_solver.keys(), scores_solver.values())
plt.title(f"Wpływ solvera (warstwy: {best_hidden}, aktywacja: {best_activation})")
plt.ylabel("Balanced Accuracy")
plt.ylim(0.7, 1.0)
plt.show()

best_solver = max(scores_solver, key=scores_solver.get)
print("Najlepszy solver:", best_solver)
