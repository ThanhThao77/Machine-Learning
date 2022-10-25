
import numpy as np 

X=np.array([[1,5,8,6,3,10,9],[200,700,800,400,100,600,500]]).T
y=np.array([[100,300,400,200,100,400,300]]).T
# print("x = \n", X)
# print("y = \n", y)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1).T

w = ((np.linalg.pinv((Xbar@(Xbar.T))))@Xbar)@y
print("Tham so w = \n",w)

print("Du doan")
p = np.array([[1,5,700]])
print('f(x) = ',(p@w)[0][0])