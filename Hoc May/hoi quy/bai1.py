import pandas as pd
import numpy as np
from sklearn import datasets, linear_model

excel=pd.read_csv('bt1.csv') #Doc du lieu vao 
X=np.array([excel['Chi phi'],excel['Dan so']]).T #Tim ma tran X
y=np.array([excel['Doanh thu']]).T #Tim ma tran y

#Cach 1: Goi thu vien sklearn
regr = linear_model.LinearRegression()  #Khai bao doi tuong regr la hoi quy tuyen tinh
regr.fit(X,y) #Truyen du lieu X va y cho doi tuong regr
print( 'w = ', regr.coef_ ) #In ma tran he so hoi quy tim duoc
print("Du doan:",regr.predict(np.array([[6,400]]))) #Du doan 

#Cach 2: Thuc hien nhan ma tran
one = np.ones((X.shape[0], 1)) #Tao ma tran 1
Xbar = np.concatenate((one, X), axis = 1).T #Ghep ma tran 1 voi X
w = ((np.linalg.pinv((Xbar@(Xbar.T))))@Xbar)@y #Tim ma tran he so hoi quy w
p=np.array([[1,6,400]]) #Khai bao du lieu du doan
print('Dự đoán: ',(p@w)[0][0]) #Du bao



