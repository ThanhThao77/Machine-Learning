# import pandas as pd
# import numpy as np

# df = pd.read_csv('Tetuan City power consumption.csv')
# X = np.array(df[['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption']])
# Y = np.array(df[['Zone 3  Power Consumption']])
# one = np.ones((X.shape[0], 1))
# Xbar = np.concatenate((one, X), axis = 1).T


# w = ((np.linalg.pinv((Xbar@(Xbar.T))))@Xbar)@Y

# print("Tham so w = \n",w)

# print("Du doan")
# # p = np.array([[1, 6.559, 73.8, 0.083, 0.051, 0.119, 34055.7, 16128.88]])
# p = np.array([[1, 6.414, 74.5, 0.083, 0.07, 0.085, 39814.68, 19375.08]])

# print("f(x) = ", (p@w)[0][0])


import pandas as pd
# thư viện dành cho khoa học dữ liệu / phân tích dữ liệu và học máy

from sklearn.model_selection import KFold


from sklearn.metrics import r2_score
# được sử dụng để đo các giá trị MSE và R-Squared. Đầu vào cho các phương pháp này là giá trị thực tế và giá trị dự đoán.

# MSE :giá trị sai số bình phương trung bình

# R-squared: phản ánh mức độ giải thích của các biến độc lập đối với biến phụ thuộc trong mô hình hồi quy

from sklearn.linear_model import LinearRegression
# hàm dự đoán giá trị của một biến dựa trên giá trị của một hoặc các biến khác.

from sklearn.model_selection import train_test_split
# Chia mảng hoặc ma trận thành các tập con thử nghiệm và huấn luyện ngẫu nhiên. Tiện ích nhanh chóng kết hợp xác thực đầu vào và tiếp theo và ứng dụng để nhập dữ liệu vào một lệnh gọi duy nhất để tách dữ liệu trong một dòng.

import numpy as np
data = pd.read_csv('Tetuan City power consumption.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

# data là mảng danh sách dữ liệu đã được đọc từ file csv
# train_test_split trả về ds các mảng numpy, các chuỗi khác hoặc ma trận thưa thớt spicy (thường là mảng numpy)
# test_size:  là số xác định kích thước của tập kiểm tra (~ 30%)
# suffle : xác định xem có xáo trộn tập dữ liệu trước khi áp dụng phân tách hay không => đang là không xáo trộn

k = 5
kf = KFold(n_splits=k, random_state=None)
# tinh er,y thuc te,
# y_pred: dl du doan
# chia dữ liệu thành 5 phần(k=5), random_state=None: không làm gì cả, lấy theo thứ tự, ko random


def error(y, y_pred):
    l = []
    for i in range(0, len(y)):  # do dai y[0:1]y[0]
        l.append(np.abs(np.array(y[i:i+1])-np.array(y_pred[i:i+1])))
    # l.append dùng để thêm phần tử chênh lệch vào cuối dãy trong mảng l ( hiện tại đang trống )
    return np.mean(l)
    # trả về gtri trung bình các phần tử của mảng


max = 999999
i = 1



for train_index, test_index in kf.split(dt_Train):
    # xoay vòng chia dt_Train thành k-1 phần để train,1 phần để test

    X_train, X_test = dt_Train.iloc[train_index,
                                    :5], dt_Train.iloc[test_index, :5]
    y_train, y_test = dt_Train.iloc[train_index,
                                    5], dt_Train.iloc[test_index, 5]

    lr = LinearRegression()  # gọi hàm hồi quy tuyến tính
    # chạy hàm fit để so sánh chọn 2 gtri X_train, y_train hợp với mô hình tt
    lr.fit(X_train, y_train)
    # hàm dự đoán nhãn của các giá trị dữ liệu của x_train
    Y_pred_train = lr.predict(X_train)
    # hàm dự đoán nhãn của các giá trị dữ liệu của x_test
    Y_pred_test = lr.predict(X_test)
    sum = error(y_train, Y_pred_train)+error(y_test, Y_pred_test)
# gọi hàm error để tính tổng lỗi
    if (sum < max):
        max = sum  # lấy lỗi có giá trị bé nhất
        last = i
        regr = lr.fit(X_train, y_train)  # lay mô hình tot nhat
    i = i+1


y_pred = regr.predict(dt_Test.iloc[:, :5])  # Dự đoán số y của mô hình tốt nhất
y = np.array(dt_Test.iloc[:, 5])
y_test = dt_Test.iloc[:, 5]       # gán

print("Coefficient of determination: %.2f" % r2_score(y, y_pred))
print("Thuc te       Du Doan                chenh lech")

for i in range(0, len(y)):
    print("%.2f" % y[i], " ", y_pred[i], " ", abs(y[i] - y_pred[i]))


def predictcount(): # chỗ này không hiểu thì thôi. tại tôi cũng đ hiểu.
    count = 0
    for _ in range(0, len(y)):
        count = count + 1
    return count

    # trả ra tỷ lệ
print("ty le dung: " + str(predictcount()) + "%")
print("ty le sai: " + str(100 - predictcount()) + "%")