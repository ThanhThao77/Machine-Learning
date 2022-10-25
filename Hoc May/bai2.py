import numpy as np
# Bài 2
X = np.array([[60,2,10],[40,2,5],[100,3,7]]).T
Y = np.array([10,12,20])
print("X^T =\n", X)
print("Y = ", Y)


A = np.array(np.linalg.pinv(X @ X.T))
B = np.array((A @ X@Y))
print("Tham so W(Cach khac) = \n",B)
# w = Gia Nghich Dao (X @ X.T) * X @ y
w = (np.linalg.pinv(X @ X.T) @X@Y)
print("Tham so w = \n",w)

_x = np.array([50,2,8])
print("f(x) = ", w@_x)