# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:41:45 2021

@author: KBCI
"""

#%% Data Split

# data = pd.read_csv("used_car.csv")
# data = data.set_index('id')

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, shuffle=True, test_size=0.2, random_state=42)

X_train = train_set.drop('price', axis=1)
y_train = train_set['price'].copy()
X_test = test_set.drop('price', axis=1)
y_test = test_set['price'].copy()


#%% Standard Scaling

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
dat = std_scaler.fit_transform(data)


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
print(rnd_search.best_estimator_b)

rnd_predictions = rnd_search.predict(X_test)
forest_mse = mean_squared_error(y_test, rnd_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

## importances visualization
import matplotlib as mpl
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

importances = pd.Series(rnd_search.feature_importances_, f_prepared.columns)
importances = importances.sort_values()
importances[-30:].plot.barh()
plt.yticks(rotation = 45)


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

svm_search.fit(X, y)
print(svm_search.best_params_)
print(svm_search.best_estimator_b)

svm_predictions = svm_rnd_search.predict(X_test)


#%% XGBoost

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from xgboost import XGBRegressor


xgb_reg =XGBRegressor(objective='reg:squarederror', verbose=False)
xgb_param_grid = {'max_depth': [2,3,5,7], 
              'subsample': [0.3, 0.5, 0.7],
              'n_estimators': [50, 100, 150]}

xgb_search = GridSearchCV(estimator=xgb_reg, scoring=make_scorer(mean_squared_error, squared=False),
                   param_grid= xgb_param_grid, 
                   cv=10,
                   verbose=False)

xgb_search.fit(X_train, y_train)

print(xgb_search.estimator)
print(xgb_search.best_score_)


from sklearn.metrics import mean_squared_error

xgb_predictions = xgb_search.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
xgb_rmse