import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from quantmod.timeseries import *
from quantmod.indicators import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, 
                                    RandomizedSearchCV, 
                                    TimeSeriesSplit, 
                                    )

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, to_graphviz

from sklearn.metrics import (accuracy_score,
                             auc,
                             roc_curve, 
                             RocCurveDisplay,
                             ConfusionMatrixDisplay,
                             confusion_matrix
                            )

from sklearn.metrics import (classification_report, 
                             confusion_matrix 
                            )


df = pd.read_csv('https://github.com/mvannucc/ml-algorithms/blob/main/dataset/spy.csv', 
                 index_col=0, 
                 parse_dates=True, 
                 dayfirst=True
                )

df['Returns'] = dailyReturn(df['Adj Close'])
print(df.describe().T)

features_list = []

for r in range(5, 50, 5):
    df['SMA_'+str(r)] = SMA(df['Adj Close'], r)  
    df['EMA_'+str(r)] = EMA(df['Adj Close'], r)  
    features_list.append('SMA_'+str(r))
    features_list.append('EMA_'+str(r))

df['ATR'] = ATR(df)  
df['BBANDS_L'] = BBands(df['Adj Close'],5,2)[0]  
df['BBANDS_M'] = BBands(df['Adj Close'],5,2)[1]  
df['BBANDS_U'] = BBands(df['Adj Close'],5,2)[2]  
df['RSI'] = RSI(df['Adj Close'], 14)  
df['MACD'] = MACD(df['Adj Close'], 5, 12, 26)[0]

df['Target'] = np.where(df['Adj Close'].shift(-1) > 0.995*df['Adj Close'], 1, 0)
df.dropna(inplace=True)

X = df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Returns', 'Target'], axis=1)
y = df['Target']

print(np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rebalancing
def cwts(dfs):    
    c0, c1 = np.bincount(dfs)    
    w0=(1/c0)*(len(df))/2    
    w1=(1/c1)*(len(df))/2     
    return {0: w0, 1: w1}

train_weights = [cwts(y_train)[0] if label == 0 else cwts(y_train)[1] for label in y_train]
dtrain = xgb.DMatrix(X_train_scaled, label=y_train, nthread=-1, weight=train_weights, feature_names=X_train.columns.to_list())

test_weights = [cwts(y_test)[0] if label == 0 else cwts(y_test)[1] for label in y_test]
dtest = xgb.DMatrix(X_test_scaled, label=y_test, nthread=-1, weight=test_weights, feature_names=X_test.columns.to_list())

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

base_model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'eval')]
)

y_prob = base_model.predict(dtest)
y_pred = np.round(y_prob)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])
disp.plot(cmap='Blues')
plt.title('base Model Confusion Matrix')
plt.tight_layout()
plt.show()

print(f"\n Base Model Classification Report:")
print(classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)  
roc_auc = auc(fpr, tpr)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp.plot()

plt.title('Base Model ROC Curve')
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

train_preds = np.round(base_model.predict(dtrain))
train_accuracy = accuracy_score(y_train, train_preds)

test_preds = np.round(base_model.predict(dtest))
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Difference (Training - Test): {train_accuracy - test_accuracy:.4f}")

# Hyperparameter tuning
xgb_classifier = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
)

tscv = TimeSeriesSplit(n_splits=5, gap=1)

param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5, 7]
}

random_search = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_grid,
    n_iter=100,
    scoring='roc_auc',
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)

print("\nBest parameters found by RandomizedSearchCV:")
print(random_search.best_params_)
print(f"Best score: {random_search.best_score_:.4f}")

optimized_params = random_search.best_params_.copy()
optimized_params['eta'] = optimized_params.pop('learning_rate') 
optimized_params['objective'] = 'binary:logistic'
optimized_params['eval_metric'] = 'logloss'

print("\nTraining optimized XGBoost model with best parameters...")
optimized_model = xgb.train(
    optimized_params,
    dtrain,
    num_boost_round=optimized_params.pop('n_estimators'),
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=20
)

y_proba = optimized_model.predict(dtest)
y_pred = np.round(y_proba)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])
disp.plot(cmap='Blues')
plt.title('Optimized Model Confusion Matrix')
plt.tight_layout()
plt.show()

print("\n Optimized Model Classification Report:")
print(classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp.plot()

plt.title('Optimized Model ROC Curve')
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

train_preds = np.round(optimized_model.predict(dtrain))
train_accuracy = accuracy_score(y_train, train_preds)

test_preds = np.round(optimized_model.predict(dtest))
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Difference (Training - Test): {train_accuracy - test_accuracy:.4f}")
