# import tensorflow and keras
import tensorflow 
from tensorflow import keras

print(f" TensorFlow version is {tensorflow.__version__} \n Keras version is {tensorflow.keras.__version__}")

# logging
import logging
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

# supress warnings
import warnings
warnings.filterwarnings("ignore")

# base libraries
import os
import random
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

# scikit-learn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# tensorflow modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, RMSprop 
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# plotting & outputs
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

from pprint import pprint

def set_seeds(seed=2025): 
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)

X = np.array([i+1 for i in range(60)])
X = np.array(X).reshape(20,3,1)

y = []
for each in X:
    y.append(each.sum())
y = np.array(y)

print(f'y: {y}')

nn = Sequential()
nn.add(LSTM(50, activation='relu', input_shape=(3, 1)))
nn.add(Dense(1))
nn.compile(optimizer='adam', loss='mse')
print(nn.summary())

nn.fit(X, y, batch_size=64, epochs=1000, validation_split=0.2, verbose=0)

test_input = np.array([70,71,72])
test_input = test_input.reshape((1, 3, 1))
test_output = nn.predict(test_input, verbose=0)
print(test_output)
