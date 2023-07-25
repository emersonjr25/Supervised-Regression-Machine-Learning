"""
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
THEME: MACHINE LEARNING SUPERVISIONED
"""

###### import data ######

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

data = pd.read_csv('data/data.csv')

data.columns

#other variables: Street, LotShape, ExterQual, CentralAir
variables = ['LotFrontage', 'LotArea', 'YearBuilt', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'MasVnrArea', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'PoolArea']

### only variables that is important in first moment ###
data_X = data[variables]
data_Y = data['SalePrice']

#data_train_X.shape

### isna ###
na_columns = data_X.isna().sum()
na_columns = na_columns[na_columns > 0]
na_columns
data_Y.isna().sum()

data_X = data_X.drop(na_columns.index[0], axis = 1)
data_X = data_X.fillna(data_X.mean())

#data_X.shape
#data_X.isna().sum()

data_X.describe()
data_X.dtypes

#### machine learning ####

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.3, random_state = 1)

#model 1
model1 = LinearRegression()
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)
mae1 = mean_absolute_error(y_test, prediction1)
r2_1 = r2_score(y_test, prediction1)

print('First model result mae:', mae1)
print('First model result r2:', r2_1)

#model 2
best_model = []
array = np.array([10, 50, 100, 150, 200, 300, 500, 1000, 3000])

for i in range(0, len(array)):
    model2 = RandomForestRegressor(n_estimators=array[i], random_state=1)
    model2.fit(X_train, y_train)
    prediction2 = model2.predict(X_test)
    mae2 = mean_absolute_error(y_test, prediction2)
    best_model.append(mae2)
    print(array[i], best_model)

model2 = RandomForestRegressor(n_estimators=1000, random_state=1)
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
mae2 = mean_absolute_error(y_test, prediction2)
r2_2 = r2_score(y_test, prediction2)
print('Second model result mae:', mae2)
print('Second model result r2:', r2_2)

### model 3 ###
model3 = DecisionTreeRegressor(max_depth=5)
model3.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
mae3 = mean_absolute_error(y_test, prediction3)
r2_3 = r2_score(y_test, prediction3)
print('Second model result mae:', mae3)
print('Second model result r2:', r2_3)

### model 4 ###
model4 = GradientBoostingRegressor(n_estimators=100, max_depth=5)
model4.fit(X_train, y_train)
prediction4 = model4.predict(X_test)
mae4 = mean_absolute_error(y_test, prediction4)
r2_4 = r2_score(y_test, prediction4)
print('Second model result mae:', mae4)
print('Second model result r2:', r2_4)

#model for production from best model
final_model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
final_model.fit(data_X, data_Y)
