"""
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
THEME: MACHINE LEARNING SUPERVISIONED
"""

###### import data ######

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data_train = pd.read_csv('C:/Users/emers/OneDrive/Documentos/Supervisioned-Regression-Machine-Learning-main/data/train.csv')
data_test = pd.read_csv('C:/Users/emers/OneDrive/Documentos/Supervisioned-Regression-Machine-Learning-main/data/test.csv')

data_train.columns

#other variables: Street, LotShape, ExterQual, CentralAir
variables = ['LotFrontage', 'LotArea', 'YearBuilt', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'MasVnrArea', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'PoolArea']

### only variables that is important in first moment ###
data_train_X = data_train[variables]
data_train_Y = data_train['SalePrice']

#data_train_X.shape

### isna ###
na_columns = data_train_X.isna().sum()
na_columns = na_columns[na_columns > 0]
na_columns
data_train_Y.isna().sum()

data_train_X = data_train_X.drop(na_columns.index[0], axis = 1)
data_train_X = data_train_X.fillna(data_train_X.mean())

#data_train_X.shape
#data_train_X.isna().sum()

data_train_X.describe()
data_train_X.dtypes

#### machine learning ####

X_train, X_test, y_train, y_test = train_test_split(data_train_X, data_train_Y, test_size = 0.3, random_state = 1)

#model 1
model1 = LinearRegression()
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, prediction1)
r2_1 = r2_score(y_test, prediction1)

print('First model result mse:', mse1)
print('First model result r2:', r2_1)

#model 2
model2 = RandomForestRegressor(n_estimators=100, random_state=1)
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
mse2 = mean_squared_error(y_test, prediction2)
r2_2 = r2_score(y_test, prediction2)
print('Second model result mse:', mse2)
print('Second model result r2:', r2_2)