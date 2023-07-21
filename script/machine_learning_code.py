"""
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
THEME: MACHINE LEARNING SUPERVISIONED
"""

###### import data ######
import pandas as pd
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

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

X_train, X_val, y_train, y_val = train_test_split(data_train_X, data_train_Y, test_size = 0.3, random_state = 1)

