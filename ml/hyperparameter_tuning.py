import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://stats.oarc.ucla.edu/stat/data/binary.csv')

X = df.drop('admit', axis=1)
y = df['admit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipe1.fit(X_train, y_train)
y_pred = pipe1.predict(X_test)
print("KNN Predictions:", y_pred)

param_grid = {
    'knn__n_neighbors': range(1, 31),
    'knn__weights': ['uniform', 'distance'],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe1, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)
best_model = grid.best_estimator_
y_pred = grid.predict(X_test)
print("KNN Predictions (Optimized):", y_pred)

### Logistic Regression

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(solver='liblinear'))
])

pipe2.fit(X_train, y_train)
y_pred = pipe2.predict(X_test)
print("LogisticRegression Predictions:", y_pred)

base_param_grid = {
    'logistic__C': np.linspace(0.01,10,10),
    'logistic__penalty': ['l1', 'l2']
}

el_param_grid = {
    'logistic__C': np.linspace(0.01, 10, 10),
    'logistic__solver': ['saga'],  
    'logistic__penalty': ['elasticnet'],
    'logistic__l1_ratio': np.linspace(0.0, 1.0, 5)
}

param_grid_combined = {**base_param_grid, **el_param_grid}

grid_search = GridSearchCV(pipe2, param_grid_combined, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = grid_search.predict(X_test)
print("LogisticRegression Predictions (Optimized):", y_pred)
