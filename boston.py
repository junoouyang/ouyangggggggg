#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

boston=load_boston()
X=boston['data']
X.shape
names=boston['feature_names']

X= pd.DataFrame(X)
X.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT']
Y=boston['target']
Y= pd.DataFrame(Y)
Y.columns = ['PRICE']

data = pd.concat( [X,Y],axis=1)
data.info()
print(data.describe())
cor =np.round(data.corr(method='pearson'),2)
print(cor)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
clf=LinearRegression().fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
Y_pred=np.round(Y_pred,2)

from sklearn.metrics import mean_absolute_error,r2_score
mean_absolute_error(Y_test,Y_pred)
r2_score(y_true=Y_test,y_pred=Y_pred)

Y_test.shape
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.plot(range(102),Y_pred,color='r')
plt.plot(range(102),Y_test,color='b')
plt.title('price_test and price_predict',fontsize=20)
plt.show()

print(clf.intercept_)
clf.coef_
clf.coef_1 =pd.DataFrame(clf.coef_) 
np.round(clf.coef_1,2)
clf.coef_1.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT']
print(clf.coef_1)



