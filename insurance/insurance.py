# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:49:38 2020

@author: sarav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_set = pd.read_csv("insurance.csv")
x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,6].values

data_set.isnull().sum()
#Identify counts of catagorical variables;
for i in range(0,data_set.shape[1]):
    print(data_set.iloc[:,i].value_counts())
    print("_________________________________")
    
#label encoder
from sklearn.preprocessing import LabelEncoder
le_x = LabelEncoder()

for i in range(0,x.shape[1]):#column = x.shape[1]
    if type(x[0,i]) == str:
        x[:,i] = le_x.fit_transform(x[:,i])
    
#onehotencoder
from sklearn.preprocessing import OneHotEncoder
ohe_x =OneHotEncoder(categorical_features=[5]) 
x = ohe_x.fit_transform(x).toarray()       
    
#dummy variable trap
x =x[:,1:]

#split the data as train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=0)

#linear regression
from sklearn.linear_model import LinearRegression
regress =LinearRegression()
regress.fit(x_train,y_train)

#prediction
y_pred = regress.predict(x_test)
regress.score(x_train,y_train)
regress.score(x_test,y_test)

#backward elination
import statsmodels.api as sm
x = np.append(arr=np.ones(shape = (1338,1),dtype=int),values=x,axis=1)

x_ov = x[:,[0,1,2,3,4,5,6,7,8]]
regress_ols= sm.OLS(endog =y,exog =x_ov).fit()
regress_ols.summary() 


#iteration2
x_ov=x[:,[0,1,2,3,4,6,7,8]]
regress_ols= sm.OLS(endog =y,exog =x_ov).fit()
regress_ols.summary() 



#iteration3
x_ov=x[:,[0,2,3,4,6,7,8]]
regress_ols= sm.OLS(endog =y,exog =x_ov).fit()
regress_ols.summary() 



#iteration4
x_ov=x[:,[0,2,4,6,7,8]]
regress_ols= sm.OLS(endog =y,exog =x_ov).fit()
regress_ols.summary() 



#iteration5
x_ov=x[:,[0,4,6,7,8]]
regress_ols= sm.OLS(endog =y,exog =x_ov).fit()
regress_ols.summary() 


x_ovc = x_ov[:,1:]
x_train_ov ,x_test_ov,y_train_ov ,y_test_ov =train_test_split(x_ovc,y,test_size =.30,random_state=0)
regress_ov3 = LinearRegression()
regress_ov3.fit(x_train_ov,y_train_ov)
regress_ov3.score(x_train_ov,y_train_ov)
regress_ov3.score(x_test_ov,y_test_ov)


