# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:33:07 2020

@author: Harish
"""

import pandas as pd 
import numpy as np 
import seaborn as sns
test=pd.read_csv("SalaryData_Test.csv")
train=pd.read_csv("SalaryData_Train.csv")
test['Salary'].unique()
test.Salary.value_counts()
from sklearn import preprocessing
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
for i in string_columns:
    number=preprocessing.LabelEncoder()
    train[i]=number.fit_transform(train[i])
    test[i]=number.fit_transform(test[i])
    
    
train_X = train.iloc[:,0:13]
train_y = train.iloc[:,13]
test_X  = test.iloc[:,0:13]
test_y  = test.iloc[:,13]


from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)


model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y)
#77.79 Accuracy

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)
#79.64 accuracy