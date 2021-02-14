# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:10:25 2020

@author: Harish
"""

import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.svm import SVC
forest=pd.read_csv("forestfires.csv")

string_columns=["month","day",]
for i in string_columns:
    number=preprocessing.LabelEncoder()
    forest[i]=number.fit_transform(forest[i])
    
from sklearn.model_selection import train_test_split
train,test = train_test_split(forest,test_size = 0.3,random_state=0)


train_x=train.iloc[:,0:30]
train_y=train.iloc[:,30]
test_x=test.iloc[:,0:30]
test_y=test.iloc[:,30]

model_linear=SVC(kernel="linear")
model_linear.fit(train_x,train_y)
pred_linear_test=model_linear.predict(test_x)
np.mean(pred_linear_test==test_y)
#accuracy=96.79

model_poly=SVC(kernel="poly")
model_poly.fit(train_x,train_y)
pred_poly_test=model_poly.predict(test_x)
np.mean(pred_poly_test==test_y)
#accuracy=75.64

model_rbf=SVC(kernel="rbf")
model_rbf.fit(train_x,train_y)
pred_rbf_test=model_rbf.predict(test_x)
np.mean(pred_rbf_test==test_y)
#accuracy=72.435


#Hence the best model is Linear model