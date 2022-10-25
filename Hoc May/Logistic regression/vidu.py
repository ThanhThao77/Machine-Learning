#Lap trinh = pp giam dao ham de tim cuc tieu
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(2)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])  #X: thời gian ôn thi
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])  # tỉ lệ đỗ trượt

# extended data 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

def sigmoid(s): 
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000): #định nghĩa hàm tìm cực trị = pp giảm đạo hàm
    w = [w_init]    # w gán bằng gtri ban đầu
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count: #số lần lặp nhỏ hơn số lần lặp tối đa
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)  #xi lấy trong mảng x
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))  #zi tính theo công thức
            w_new = w[-1] + eta*(yi - zi)*xi  #thực hiện công thức cập nhật w mới = w cũ + 1 hệ số
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w #trả về điểm cực trị tìm đc (hệ số hồi quy tìm đc)

eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1) #cho w khởi đầu là đại lượng ngẫu nhiên

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])

#dự đoán sv ôn bài với số giờ x = 5
dudoan = np.array([1,5])
print("Du doan: ", sigmoid(w[-1].T @ dudoan))
#gtri đầu ra chạy từ 0 - 1: trên 0.5 là đỗ, dưới 0.5 là trượt. Phù hợp với bài toán xác suất, dùng để làm bài phân lớp

