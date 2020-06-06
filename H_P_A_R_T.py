
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing train set
train_set = pd.read_csv('train.csv')

train_set.shape
#train_set.select_dtypes(include='int64').fillna(method='ffilll')
# train_set.head() # 81 columns

train_set.columns # taking names of the columns
train_set.dtypes # int64, float64
train_set.select_dtypes(include='object').columns # determining variable that needs dummy variables
train_set.select_dtypes(include='object')
# Dummy varibles
train_set = pd.get_dummies(train_set, columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                                   'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                                   'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                                   'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                   'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                                   'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                                   'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                                   'SaleType', 'SaleCondition'],drop_first=True)
train_set.shape
#       #       #       #       #
# train_set.columns.get_loc('SalePrice') # find index of the SalePrice
X = train_set.drop('SalePrice',axis = 1).iloc[:,1:-1].values
y = train_set.iloc[:, 37].values
#
# Taking care of missing numerical data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:,:] = imputer.transform(X[:, :])
# split into trainin set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# train = train_set.sample(frac=0.80,random_state = 123)
# # test = train_set.loc[~train_set.index.isin(train.index),:]

# -------------------------------------------------------------------Fitting the Regression Model to the dataset
X_train = X
y_train = y
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X_train,y_train)
# Predicting a new result
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(np.log(y_test), np.log(y_pred)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing test set
test_set = pd.read_csv('test.csv')
test_set.shape
train_set = train_set.drop('SalePrice',axis = 1)
test_set = pd.get_dummies(test_set, columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                                   'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                                   'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                                   'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                   'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                                   'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                                   'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                                   'SaleType', 'SaleCondition'],drop_first=True)
#
feature_difference = set(train_set) - set(test_set)
feature_difference_df = pd.DataFrame(data=np.zeros((test_set.shape[0], len(feature_difference))),
                                     columns=list(feature_difference))
test_set = test_set.join(feature_difference_df)
test_set.shape

X = test_set.iloc[:,1:-1].values
X.shape
# Taking care of missing numerical data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:,:] = imputer.transform(X[:, :])
#


y_pred = regressor.predict(X)
y_pred.shape
test_set.iloc[:,1]

#   Missing data handling
# DataFrame.dropna(self[, axis, how, thresh, …])	#Remove missing values.
# DataFrame.fillna(self[, value, method, …])	#Fill NA/NaN values using the specified method.
# DataFrame.replace(self[, to_replace, value, …])	#Replace values given in to_replace with value.
# DataFrame.interpolate(self[, method, axis, …])	#Interpolate values according to different methods.

pd.DataFrame({'Id':test_set.iloc[:,0],'SalePrice':y_pred}).set_index('Id').to_csv('sub.csv')