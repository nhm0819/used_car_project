# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:41:45 2021

@author: NHM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% Data Split

data = pd.read_csv("used_car.csv")
data = data.set_index('id')

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, shuffle=True, test_size=0.2, random_state=42)

X_train = train_set.drop('price', axis=1)
y_train = train_set['price'].copy()
X_test = test_set.drop('price', axis=1)
y_test = test_set['price'].copy()


#%% Standard Scaling

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
data = std_scaler.fit_transform(data)


#%% Random Forest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid = [
    {'n_estimators': [100, 200, 300], 'max_features': [7,8,9]},
    {'bootstrap': [False], 'n_estimators': [100, 200, 300], 'max_features':[7,8,9]}
]

rnd = RandomForestRegressor(random_state = 42)
rnd_search = GridSearchCV(rnd, param_grid, cv=10,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
rnd_search.fit(X_train, y_train)

print(rnd_search.best_params_)
# print(rnd_search.best_estimator_b)
rnd_train_predictions = rnd_search.predict(X_train)
forest_train_mse = mean_squared_error(y_train, rnd_train_predictions)
forest_train_rmse = np.sqrt(forest_train_mse)
forest_train_rmse

rnd_predictions = rnd_search.predict(X_test)
forest_mse = mean_squared_error(y_test, rnd_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

### train_rmse : 717.61
### test_rmse : 4840.39  --> overfitting

## importances visualization
import matplotlib as mpl
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

importances = pd.Series(rnd_search.feature_importances_, y_test)
importances = importances.sort_values()
importances[-30:].plot.barh()
plt.yticks(rotation = 45)

import joblib
joblib.dump(rnd_search, 'rnd_model.pkl') 

# model  : {'bootstrap': False, 'max_depth': 7, 'max_features': 7, 'n_estimators': 200} --> rmse = 8000, not overfitting
# model  : {'bootstrap': False, 'max_depth': 11, 'max_features': 7, 'n_estimators': 200} --> rmse = 8000, not overfitting
# model2 : {'bootstrap': False, 'max_depth': 15, 'max_features': 8, 'n_estimators': 300} --> rmse = 7000, not overfitting
# model3 : 

######### Fine tuning

rnd2 = RandomForestRegressor(random_state = 42)

param_grid = [
    {'bootstrap': [False], 'n_estimators': [300],
     'max_features':[7,8,9], 'max_depth':[15, 18, 21] }
]

rnd_search2 = GridSearchCV(rnd2, param_grid, cv=10,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)

rnd_search2.fit(X_train, y_train)

rnd_train_predictions2 = rnd_search2.predict(X_train)
forest_train_mse2 = mean_squared_error(y_train, rnd_train_predictions2)
forest_train_rmse2 = np.sqrt(forest_train_mse2)
forest_train_rmse2

rnd_predictions2 = rnd_search2.predict(X_test)
forest_mse2 = mean_squared_error(y_test, rnd_predictions2)
forest_rmse2 = np.sqrt(forest_mse2)
forest_rmse2

import joblib
joblib.dump(rnd_search2, 'rnd_model2.pkl') 

# a = joblib.load('rnd_model.pkl')
# a.best_params_

#%% SVM

# kernel trick
# kernel = 'poly'
# (gamma * < x, x'> + coef0)^degree
# gamma = 'scale' : 1/(num of attribs + data var)
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

svm_reg = SVR()

param_distribs = {'kernel': ['linear', 'rbf', 'poly'],
                  'C': reciprocal(20, 20000),
                  'gamma': expon(scale=1.0)}

svm_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                cv=10, n_iter=20,
                                scoring="neg_mean_squared_error",
                                random_state=42)

svm_search.fit(X_train, y_train)
print(svm_search.best_params_)
print(svm_search.best_estimator_b)

svm_predictions = svm_search.predict(X_test)


#%% XGBoost

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from xgboost import XGBRegressor


xgb_reg =XGBRegressor()

xgb_reg =XGBRegressor(objective = 'reg:squarederror')

xgb_param_grid = {'n_estimators':[100, 500],
                  'max_depth': [5,7,9,11,13,15], 
                  'subsample': [0.5, 0.7, 0.8],
                  'colsample_bytree': [0.6, 0.7, 0.8],
                  'learning_rate': [0.05, 0.1],
                  'min_child_weight':[2, 4, 8, 16],
                  'colsample_bytree': [0.6, 0.8],
                    'early_stopping_rounds':[30]}

xgb_search = GridSearchCV(estimator=xgb_reg, scoring='neg_mean_squared_error',
                   param_grid= xgb_param_grid, 
                   cv=5,
                   verbose=True)

xgb_search.fit(X_train, y_train)

import joblib
joblib.dump(xgb_search, 'xgb_model.pkl') 
joblib.dump(xgb_search.best_params, './xgb_best_params.pkl')
joblib.dump(xgb_search.best_estimators_b, './xgb_best_estimator_.pkl')

print(xgb_search.estimator)
print(xgb_search.best_score_)


from sklearn.metrics import mean_squared_error

xgb_predictions = xgb_search.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
xgb_rmse