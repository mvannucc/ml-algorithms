import pandas as pd
import numpy as np
import random

# SOM & Sklearn library
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Visualization library
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
plt.style.use('fivethirtyeight')

random.seed(2025)
np.random.seed(2025)

df = pd.read_csv('https://github.com/mvannucc/ml-algorithms/blob/main/dataset/cluster.csv')

X = df[['Returns', 'Volatility']]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

som = MiniSom(5, 5, 2, learning_rate=0.5, sigma=5, random_seed=42)
som.random_weights_init(X)
som.train_batch(X, 10000, verbose=True)

for ix in range(len(X)):
    winner = som.winner(X[ix])
    plt.text(winner[0], winner[1], df.Companies[ix], bbox=dict(facecolor='white', alpha=0.5, lw=0)) 
plt.imshow(som.distance_map())
plt.grid(False)
plt.title('Self Organizing Maps')
plt.show()

for ix in range(len(X)):
    winner = som.winner(X[ix])
    print(winner[0], winner[1], df['Companies'][ix])
