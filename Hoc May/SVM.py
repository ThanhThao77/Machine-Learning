from sklearn import svm
import pandas as pd
import numpy as np

df = pd.read_csv('SVM.csv')
d = {'Nam': 1, 'Nữ': 0}
df['Giới tính'] = df['Giới tính'].map(d)

d = {'THCS': 0, 'THPT': 1}
X = [[1, 1], [1, 2], [1.5, 1.5], [3, 3], [4, 4], [4.5, 3.5]]
y = [-1, -1, -1, 1, 1, 1]

clf = svm.SVC(kernel = 'linear')

clf.fit(X, y)
w = clf.coef_
b = clf.intercept_

print('w = ', w)
print('b = ', b)
print("Du bao:")
kq= clf.predict([[2.5, 20.2]])
# #kq= clf.predict([[-2., -2.]])
print(kq)