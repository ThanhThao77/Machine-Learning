import pandas as pd
import numpy as np

df = pd.read_csv('accelerometer.csv')
X = np.array(df[['wconfid', 'pctid', 'x', 'y']])
Y = np.array(df[['z']])
one = np.ones((X.shape[0], 1))
Xnew = np.concatenate((one, X), axis = 1).T

# print(Y)
w = (np.linalg.pinv(Xnew @ (Xnew.T)) @Xnew)@Y
print("Tham so w = \n",w)

print("Du doan")
p = np.array([[1,1, 20, 1.004, 0.09]])

print("f(x) = ", (p@w)[0][0])

# TẬP DỮ LIỆU GIA TỐC KẾ
# Có 5 thuộc tính trong tập dữ liệu: wconfid, pctid, x, y và z.
# wconfid: ID Cấu hình Trọng lượng (1 - 'đỏ' - cấu hình bình thường; 2 - 'xanh lam' - cấu hình vuông góc; 3 - 'xanh lá cây' - cấu hình ngược lại)
# pctid: ID Phần trăm Tốc độ RPM của Quạt làm mát (20 nghĩa là 20%, v.v. ).
# x: Giá trị x của gia tốc kế.
# y: Giá trị y của gia tốc kế.
# z: Giá trị z của gia tốc kế.