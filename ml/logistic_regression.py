import pandas as pd
import numpy as np
from quantmod.timeseries import *

import matplotlib.pyplot as plt

# Classifier
from sklearn.linear_model import LogisticRegression

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
                                    train_test_split, 
                                    TimeSeriesSplit,
                                    GridSearchCV
                                    )

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
                            accuracy_score,
                            f1_score,
                            log_loss,
                            RocCurveDisplay,
                            ConfusionMatrixDisplay,
                            classification_report
                            )

df = pd.read_csv(
    'https://github.com/mvannucc/ml-algorithms/blob/main/dataset/nifty.csv',
    index_col=0, 
    parse_dates=True, 
    dayfirst=True
    )

plt.plot(df['Close'])
#plt.show()

print(df.describe().T)
print(df.isnull().sum())

df['O-C'] = df.Open - df.Close
df['H-L'] = df.High - df.Low

X = df[['O-C', 'H-L']].values

y = np.where(df['Close']>df['Close'].shift(1),1,0)

print(pd.Series(y).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Train and Test Size {len(X_train)}, {len(X_test)}")

basemodel = Pipeline([
    ("scaler", StandardScaler()), 
    ("classifier", LogisticRegression(
        class_weight='balanced')) 
]) 

basemodel.fit(X_train, y_train)

y_pred  = basemodel.predict(X_test)
y_proba = basemodel.predict_proba(X_test)

print(y_pred[-20:])
y_proba[-20:]

acc_train = accuracy_score(y_train, basemodel.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)

print(f'Train Accuracy: {acc_train:0.4}, Test Accuracy: {acc_test:0.4}')

disp = ConfusionMatrixDisplay.from_estimator(
        basemodel,
        X_test,
        y_test,
        cmap=plt.cm.Blues
    )
plt.title('Confusion matrix')
plt.show()

print(classification_report(y_test, y_pred))

disp = RocCurveDisplay.from_estimator(
            basemodel, 
            X_test, 
            y_test,
            name='Baseline Model')
plt.title("AUC-ROC Curve \n")
plt.plot([0,1],[0,1],linestyle="--", label='Random 50:50')
plt.legend()
plt.show()

### Hyperparameters tuning ###

basemodel.get_params()
tscv = TimeSeriesSplit(n_splits=5, gap=1)

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10]
    }

grid_search = GridSearchCV(
    estimator=basemodel, 
    param_grid=param_grid, 
    scoring='roc_auc',
    n_jobs=-1,
    cv=tscv, 
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

acc_train = accuracy_score(y_train, best_model.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)

print(f'\n Training Accuracy \t: {acc_train :0.4} \n Test Accuracy \t\t: {acc_test :0.4}')

disp = ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        cmap=plt.cm.Blues
    )
plt.title('Confusion matrix')
plt.show()

disp = RocCurveDisplay.from_estimator(
            best_model, 
            X_test, 
            y_test,
            name='Tuned Logistic')
plt.title("AUC-ROC Curve \n")
plt.plot([0,1],[0,1],linestyle="--", label='Random 50:50')
plt.legend()
plt.show()

print(classification_report(y_test, y_pred))

### Trading Strategy ###

df1 = df.copy()                              # df[-len(X_test)
df1['Signal'] = best_model.predict(X)        # tunedmodel.predict(X_test)
df1['Returns'] = np.log(df1['Close']).diff().fillna(0)
df1['Strategy'] =  df1['Returns'] * df1['Signal'].shift(1).fillna(0)
df1.index = df1.index.tz_localize('utc')

print(df1)
