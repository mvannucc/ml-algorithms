# Data Manipulation
import numpy as np
import pandas as pd

# Visualizaiton
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Regressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# Metrics
from sklearn.metrics import mean_squared_error

boston_data = pd.read_csv('https://github.com/mvannucc/ml-algorithms/blob/main/dataset/boston.csv') 

X = boston_data.drop('medv', axis=1)      
y = boston_data['medv']                    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Train and Test Size {len(X_train)}, {len(X_test)}")

#########################
### Linear Regression ###
#########################

pipe = Pipeline([
    ('scaler', StandardScaler()),       
    ('regressor', LinearRegression())   
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'MSE: {mse:0.4}')    
print(f'RMSE: {rmse:0.4}')  

print('\n Linear regression')
print(f'R^2 Train: {pipe.score(X_train, y_train):0.4}')
print(f'R^2 Test: {pipe.score(X_test, y_test):0.4}')

pipe['regressor'].coef_
pipe['regressor'].intercept_

##############################
### Regularized Regression ###
##############################

# LASSO

laso = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Lasso(alpha=0.1))
])

laso.fit(X_train, y_train)
print('\n Lasso method')
print(f'R^2 Train: {laso.score(X_train, y_train):0.4}')
print(f'R^2 Test: {laso.score(X_test, y_test):0.4}')

alpha_range = np.linspace(0.01,1,50)
score = []
la_mse = []
la_rmse = []

for i in alpha_range:
    laso = Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(alpha=i))])
    laso.fit(X_train, y_train)
    mse = mean_squared_error(y_test, laso.predict(X_test))
    rmse = np.sqrt(mse)
    
    la_mse.append(mse)
    la_rmse.append(rmse)
    score.append(laso.score(X_test, y_test))

fig, ax = plt.subplots(1,2, figsize=(20,8))
ax[0].plot(alpha_range, la_rmse, 'orange')
ax[0].set_title('Root Mean Square Error')
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('RMSE')

ax[1].plot(alpha_range, score, 'r-')
ax[1].set_title('Coefficient of Determination')
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('R-Squared')

plt.show()

# RIDGE

rid = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1))
])

rid.fit(X_train, y_train)
print('\n Ridge method')
print(f'R^2 Train: {rid.score(X_train, y_train):0.4}')
print(f'R^2 Test: {rid.score(X_test, y_test):0.4}')

alpha_range = np.arange(1,2000,100)
score = []
rid_mse = []
rid_rmse = []

for i in alpha_range:
    rid = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=i))])
    rid.fit(X_train, y_train)   
    mse = mean_squared_error(y_test, rid.predict(X_test))
    rmse = np.sqrt(mse)

    rid_mse.append(mse)
    rid_rmse.append(rmse)
    score.append(rid.score(X_test, y_test))

fig, ax = plt.subplots(1,2, figsize=(20,8))
ax[0].plot(alpha_range, rid_rmse, 'orange')
ax[0].set_title('Root Mean Square Error')
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('RMSE')

ax[1].plot(alpha_range, score, 'r-')
ax[1].set_title('Coefficient of Determination')
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('R-Squared')

plt.show()

# ElasticNet

elastic = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))
])

elastic.fit(X_train, y_train)
print('\n ElasticNet method')
print(f'R^2 Train: {elastic.score(X_train, y_train):0.4}')
print(f'R^2 Test: {elastic.score(X_test, y_test):0.4}')

alpha_range = np.linspace(0.01,15,150)
score = []
elastic_mse = []
elastic_rmse = []
elastic_coef = []

for i in alpha_range:
    elastic = Pipeline([('scaler', StandardScaler()), ('regressor', ElasticNet(alpha=i))])
    elastic.fit(X_train, y_train)
    mse = mean_squared_error(y_test, elastic.predict(X_test))
    rmse = np.sqrt(mse)
    
    elastic_mse.append(mse)
    elastic_rmse.append(rmse)
    elastic_coef.append(elastic['regressor'].coef_)
    score.append(elastic.score(X_test, y_test))

fig, ax = plt.subplots(1,2, figsize=(20,8))
ax[0].plot(alpha_range, la_rmse, 'orange')
ax[0].set_title('Root Mean Square Error')
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('RMSE')

ax[1].plot(alpha_range, score, 'r-')
ax[1].set_title('Coefficient of Determination')
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('R-Squared')

plt.show()

