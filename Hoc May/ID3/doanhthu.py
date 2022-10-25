from csv import excel
from sklearn import linear_model
import numpy as np
import pandas as pd
excel = pd.read_csv('Doanhthu.csv')
X = np.array([excel['chiphi']], [excel['danso']]).T
Y = np.array([excel['doanhthu']]).T


print(X)
print(Y)
# reg = linear_model.LinearRegression()
# reg.fit(X, y)

# print(reg.coef_)

# X = X.T
# w = np.linalg.pinv(X @ X.T) @ X @ y
# print(w)
