import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
N_train_points = 150
X_pos = np.random.normal(0.5, 0.4, (N_train_points,2))
X_neg = np.random.normal(-0.5, 0.4, (N_train_points,2))
X_train = np.vstack((X_pos, X_neg))

y_train = ['Red'] * N_train_points + ['Blue'] * N_train_points

df = pd.DataFrame(X_train, columns=['feature1', 'feature2'])
df['label'] = y_train

N_test_points = 20
X_test = np.random.uniform(-1.2, 1.2, (N_test_points,2))

KNN = 5

distances = cdist(X_test, X_train, metric='euclidean')          
nearest_indices = np.argsort(distances, axis=1)[:, :KNN]        

y_pred = []
for index in nearest_indices:
    nearest_labels = [y_train[i] for i in index]                    
    majority_vote = Counter(nearest_labels).most_common(1)[0][0]    
    y_pred.append(majority_vote)                                    


plt.figure(figsize=(8, 6))
for x, label in zip(X_train, y_train):
    plt.scatter(*x, color=label.lower(), label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

for x, label in zip(X_test, y_pred):
    plt.scatter(*x, color=label.lower(), marker='x', label='Predicted Test' if 'Predicted Test' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("After Prediction")
plt.legend()
plt.grid(True)
plt.show()

def plot_boundary(x, y, k):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, y_encoded)

    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, Z, cmap=plt.cm.bwr, linestyles='dashed', linewidths=0.5)
    plt.scatter(x[:, 0], x[:, 1], c=y_encoded, cmap=plt.cm.bwr)
    plt.title(f'KNN Decision Boundary with {k} Nearest Neighbours')
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$', rotation='horizontal')
    plt.grid(True)
    plt.show()

plot_boundary(X_train, y_train, 5)
