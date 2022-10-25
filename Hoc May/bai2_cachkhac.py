import numpy as np
X = np.array([[60,2,10],
              [40,2,5],
              [100,3,7]])
Y = np.array([[10],
              [12],
              [20]])
XT = X.T
w = (np.linalg.pinv(X @ XT) @X)@Y
print("Tham so w = \n",w)

print("Du doan gia cua can nha x = (50,2,8) la:")
p = np.array([[50, 2, 8]])
print("f(x) = ", (p@w)[0][0])
