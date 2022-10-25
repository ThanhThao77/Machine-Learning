import pandas as pd
import numpy as np

#hàm đọc file csv lấy trong thư viện pandas
df = pd.read_csv('Tetuan City power consumption.csv')
# print(df)
X = np.array(df[['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption']])
Y = np.array(df[['Zone 3  Power Consumption']])
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1).T


w = ((np.linalg.pinv((Xbar@(Xbar.T))))@Xbar)@Y

print("Tham so w = \n",w)

print("Du doan")
p = np.array([[1, 6.313, 74.5, 0.08, 0.062, 0.1, 29128.10127, 19006.68963]])
#p = np.array([[1, 6.414, 74.5, 0.083, 0.07, 0.085, 29814.68354, 19375.07599]])

print("f(x) = ", (p@w)[0][0])