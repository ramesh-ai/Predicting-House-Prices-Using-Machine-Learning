#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:39:42 2019

@author: globalizemedevelopment
"""

#Import the necessary Python Packages. 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib as plt 
import sklearn
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')

#Let us try to understand more about the data.
data.info()
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data

#Get ridding of the columns with lot of missing data
data = data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
data



#Create Data set with numerical variables
#DataFrame.select_dtypes(self, include=None, exclude=None)
num_Data = data.select_dtypes(include = ['int64', 'float64'])
num_Data

#Find out correlation with numerical features
traindata_corr = num_Data.corr()['SalePrice'][:-1]
traindata_corr
golden_feature_list = traindata_corr[abs(traindata_corr) > 0].sort_values(ascending = False)
print("Below are {} correlated values with SalePrice:\n{}".format(len(golden_feature_list), golden_feature_list))




#Create heatmap for correlated numerical variables of top 10
data_corrheatmap = num_Data.corr()
cols = data_corrheatmap.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(num_Data[cols].values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


#Understand the distribution of the Sale Price
num_Data['SalePrice'].describe()
num_Data['SalePrice'].skew()
num_Data['SalePrice'].kurtosis()

sns.distplot(num_Data['SalePrice'], color = 'b', bins = 100)

from scipy import stats
import matplotlib.pyplot as plt
res = stats.probplot(num_Data['SalePrice'], plot=plt)

num_Data.plot.scatter(x = 'GrLivArea', y = 'SalePrice')

num_Data.plot.scatter(x = 'GarageArea', y = 'SalePrice')

num_Data.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')

num_Data.plot.scatter(x = '1stFlrSF', y = 'SalePrice')

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = num_Data)

sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = num_Data)

sns.boxplot(x = 'FullBath', y = 'SalePrice', data = num_Data)

sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = num_Data)

sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = num_Data)

sns.boxplot(x = 'YearRemodAdd', y = 'SalePrice', data = num_Data)


#Delete the outliers
num_Data = num_Data.drop(num_Data[num_Data['Id'] == 1299].index)
num_Data = num_Data.drop(num_Data[num_Data['Id'] == 524].index)



#On basis of EDA we did earlier, filter out the variable we want to use for predicting the sale price
finaldata = num_Data.filter(['OverallQual','MSSubClass', 'KitchenAbvGr','OverallCond', 'GrLivArea', 'EnclosedPorch', 'GarageArea','TotalBsmtSF',  'YearBuilt', 'SalePrice'], axis = 1)


#Transform Sale Price and GrLivArea to reduce standardize the data 
finaldata['SalePrice'] = np.log(finaldata['SalePrice'])
finaldata['GrLivArea'] = np.log(finaldata['GrLivArea'])

#Find out the columns which are missing in final data 
total = finaldata.isnull().sum().sort_values(ascending=False)
percent = (finaldata.isnull().sum()/finaldata.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



X = finaldata.iloc[:, :-1].values
y = finaldata.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#Calculate score for the Linear Regression model
regressor.score(X_train,y_train)


#Predict Value of the house using Linear Regression
ytrainpred = regressor.predict(X_train)

#Predict Value of the house on test data set 
ytestpred = regressor.predict(X_test)


#Train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)
model.fit(X_train, y_train)


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test))


