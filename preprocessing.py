# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:40:50 2021

@author: NHM
"""

import pandas as pd


#%%  Data Preprocessing

raw_data = pd.read_csv("vehicles.csv")
data = raw_data.drop(columns=['url', 'region_url', 'image_url', 'description', 'county', 'size', 'lat', 'long'])
data['VIN_counts'] = data['VIN'].map(data['VIN'].value_counts(dropna=False))
data.loc[data.VIN_counts==164081,'VIN_counts'] = 0      # VIN==0 의 value_counts는 164081개이기 때문에 이 값들은 0으로 변경.
# data = data.set_index('id')
data = data.dropna(subset=['model', 'paint_color'])     # model과 paint_color는 예측할 수 없는 부분이기 때문에 결측치 제거
data = data.drop(columns='VIN')
names = ['id', 'price', 'region', 'year', 'manufacturer', 'model', 'condition',
        'cylinders', 'fuel', 'odometer', 'title_status', 'transmission',
        'drive', 'type', 'paint_color', 'state', 'VIN_counts']
data = data[names]
data = data.set_index('id')

data.info()

# data = pd.read_csv("used_car.csv")
# data = data.set_index('id')

#%%  Data Imputation

from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

categorical = data.columns[data.dtypes=='object']
numerical = data.columns[data.dtypes!='object']

data[categorical] = data[categorical].apply(
    lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index))

imp_num = IterativeImputer(estimator=RandomForestRegressor(),
                            initial_strategy='mean',
                            max_iter=10, random_state=0)

imp_cat = IterativeImputer(estimator=RandomForestClassifier(), 
                           initial_strategy='most_frequent',
                           max_iter=10, random_state=0)

data[numerical] = imp_num.fit_transform(data[numerical])    # impute numerical data


# impute categorical (RAM이 많이 필요하므로 model을 기준으로 각각 impute)
impute_cat = pd.Index(['model', 'manufacturer'])
data[impute_cat] = imp_cat.fit_transform(data[impute_cat])

impute_cat = pd.Index(['model', 'cylinders'])
data[impute_cat] = imp_cat.fit_transform(data[impute_cat])

impute_cat = pd.Index(['model', 'fuel'])
data[impute_cat] = imp_cat.fit_transform(data[impute_cat])

impute_cat = pd.Index(['model', 'transmission'])
data[impute_cat] = imp_cat.fit_transform(data[impute_cat])

impute_cat = pd.Index(['model', 'drive'])
data[impute_cat] = imp_cat.fit_transform(data[impute_cat])

impute_cat = pd.Index(['model', 'type'])
data[impute_cat] = imp_cat.fit_transform(data[impute_cat])

impute_cat = pd.Index(['condition', 'title_status', 'manufacturer'])
data[impute_cat] = imp_cat.fit_transform(data[impute_cat])


# data.to_csv('used_car.csv')

#%% Remove outliers

# price outlier
quartile_1 = data['price'].quantile(0.25)
quartile_3 = data['price'].quantile(0.75)
IQR = quartile_3 - quartile_1
outliers = data[(data['price'] < (quartile_1 - 1.5 * IQR)) | (data['price'] > (quartile_3 + 1.5 * IQR))]
data = data.drop(outliers.index, axis=0)

# odometer outlier
quartile_1 = data['odometer'].quantile(0.25)
quartile_3 = data['odometer'].quantile(0.75)
IQR = quartile_3 - quartile_1
outliers = data[(data['odometer'] < (quartile_1 - 1.5 * IQR)) | (data['odometer'] > (quartile_3 + 1.5 * IQR))]
data = data.drop(outliers.index, axis=0)

# data.to_csv('used_car.csv')
