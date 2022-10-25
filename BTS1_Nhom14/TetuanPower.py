import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold

excel=pd.read_csv('Tetuan City power consumption.csv') #Doc du lieu vao 
X=np.array([excel['Temperature'],excel['Humidity'],excel['Wind Speed'],excel['general diffuse flows'],excel['diffuse flows']]).T #Tim ma tran X
y=np.array([excel['Zone 1 Power Consumption'],excel['Zone 2  Power Consumption'],excel['Zone 3  Power Consumption']]).T #Tim ma tran y

kf = KFold(n_splits=3) #chia thanh 3 phan
kf.get_n_splits(X) #chia tap dl X ra
KFold(n_splits=3, random_state=None, shuffle=True)  #áp dụng k fold để chia tập dl thành 3 phần, lay bat ky nen shuffle = true


scores = []

for train_index, test_index in kf.split(X): #đe chia nhieu tap ta dung vong for
    X_train, X_test = X[train_index], X[test_index] # tao các phan huan luyen
    y_train, y_test = y[train_index], y[test_index]
    regr = linear_model.LinearRegression()  #Khai bao doi tuong regr la hoi quy tuyen tinh
    regr.fit(X_train,y_train) #Truyen du lieu X va y cho doi tuong regr
    # print((y_test,regr.fit(X_train,y_train).predict(X_test)))
    # print('Giá trị: ', explained_variance_score(y_test,regr.fit(X_train,y_train).predict(X_test)))
    scores.append(explained_variance_score(y_test,regr.fit(X_train,y_train).predict(X_test)))
    print('So sanh:',np.mean(y_train),np.mean(y_test))

scores =np.array(scores)
# print('scores = ', scores)
print('Ti le trung bình:', scores.mean(),"Sai so trung bình: " ,scores.std())

















# print('Lấy dữ liệu 1')
# test_file=pd.read_csv('ENB2012_data_test.csv')
# X_file_test=np.array([test_file['X1'],test_file['X2'],test_file['X3'],test_file['X4'],test_file['X5'],test_file['X6'],test_file['X7'],test_file['X8']]).T #Tim ma tran X
# y_file_test=np.array([test_file['Y1'],test_file['Y2'],]).T #Tim ma tran y
# print("Du doan tep du lieu:",regr2.predict(X_file_test))