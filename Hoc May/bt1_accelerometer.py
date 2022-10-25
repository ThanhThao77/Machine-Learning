import pandas as pd
import numpy as np
from sklearn import datasets, linear_model

excel=pd.read_csv('accelerometer.csv')

X=np.array([excel['wconfid'],excel['pctid'], excel['x'], excel['y']]).T 
y=np.array([excel['z']]).T

# print(X)
# print(y)

#Cach 1: Goi thu vien sklearn
regr = linear_model.LinearRegression()  #Khai bao doi tuong regr la hoi quy tuyen tinh
regr.fit(X,y) #Truyen du lieu X va y cho doi tuong regr
print( 'w = ', regr.coef_ ) #In ma tran he so hoi quy tim duoc
print("Dự đoán cách 1:",regr.predict(np.array([[1, 20, 1.004, 0.09]])))

#Cach 2: Thuc hien nhan ma tran
one = np.ones((X.shape[0], 1)) #Tao ma tran 1
Xbar = np.concatenate((one, X), axis = 1).T #Ghep ma tran 1 voi X
w = ((np.linalg.pinv((Xbar@(Xbar.T))))@Xbar)@y #Tim ma tran he so hoi quy w
p=np.array([[1,1, 20, 1.004, 0.09]]) #Khai bao du lieu du doan
print ('w cach 2: ', w.T)
print('Dự đoán cách 2: ',(p@w)[0][0])


# TẬP DỮ LIỆU GIA TỐC KẾ
# Có 5 thuộc tính trong tập dữ liệu: wconfid, pctid, x, y và z.
# wconfid: ID Cấu hình Trọng lượng (1 - 'đỏ' - cấu hình bình thường; 2 - 'xanh lam' - cấu hình vuông góc; 3 - 'xanh lá cây' - cấu hình ngược lại)
# pctid: ID Phần trăm Tốc độ RPM của Quạt làm mát (20 nghĩa là 20%, v.v. ).
# x: Giá trị x của gia tốc kế.
# y: Giá trị y của gia tốc kế.
# z: Giá trị z của gia tốc kế.
