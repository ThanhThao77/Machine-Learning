import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

df = pd.read_csv('Telecust1-1.csv')
data = np.array(df)
dt_Train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False) 

X_train = dt_Train[:, :5]
y_train = dt_Train[:, 5]
X_test  = dt_Test[:, :5]
y_test  = dt_Test[:, 5]

#CART
tree_cart = DecisionTreeClassifier(criterion = 'gini') 
tree_cart.fit(X_train, y_train)                              # đưa ra mô hình tốt nhất của CART với dữ liệu đầu vào (X_train,y_train)
y_pred = tree_cart.predict(X_test)                           # đưa ra dự đoán của CART

#ID3
tree_id3 = DecisionTreeClassifier(criterion = 'entropy')
tree_id3.fit(X_train, y_train)
y_pred1 = tree_id3.predict(X_test)

#perceptron
pla = Perceptron()
pla.fit(X_train, y_train)
y_pred2 = pla.predict(X_test)

print("Thực tế \t Dự đoán(cart) \t Dự đoán(id3) \t Dự đoán(perceptron)" )
for i in range (0, len(y_test)):
    print( y_test[i],"\t\t",  y_pred[i],"\t\t", y_pred1[i], "\t\t", y_pred2[i])

count_cart = 0
count_id3 = 0
count_pla = 0
for i in range(0,len(y_test)):
    if(y_test[i] == y_pred[i]):
        count_cart= count_cart + 1
    if(y_test[i] == y_pred1[i]):
        count_id3 = count_id3 +1
    if(y_test[i] == y_pred2[i]):
        count_pla = count_pla + 1


print('Ty le cart du doan dung:', count_cart/len(y_pred)*100, "%")
print('Ty le cart du doan sai:',100 - count_cart/len(y_pred)*100, "%")

print(" ")

print('Ty le id3 du doan dung:', count_id3/len(y_pred1)*100, "%")
print('Ty le id3 du doan sai:',100 - count_id3/len(y_pred1)*100, "%")

print(" ")

print('Ty le perceprtron du doan dung:', count_pla/len(y_pred2)*100, "%")
print('Ty le perceptron du doan sai:',100 - count_pla/len(y_pred2)*100, "%")
