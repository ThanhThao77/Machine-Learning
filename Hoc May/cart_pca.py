import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC

data = pd.read_csv('winequality-white.csv')
X = np.array(data[["x1", "x2", "x3", "x4", "x5",
             "x6", "x7", "x8", "x9", "x10", "x11"]].values)
y = np.array(data[["y"]])
cart = DecisionTreeClassifier()

max = 0
for j in range(1, 12):
    print("lan", j)
    pca = PCA(n_components=j)
    print(pca)
    pca.fit(X)
    Xbar = pca.transform(X)  # ap dung giam kich thuoc cho X.
    X_train, X_test, y_train, y_test = train_test_split(
        Xbar, y, test_size=0.3, shuffle=False)
    cart.fit(X_train,y_train)
    pred = cart.predict(X_test)
    rate=accuracy_score(pred,y_test)
    if(max>accurary){
        num_pca = j
        pca_best = pca
        max = rate
        modelmax = svc
    }
print("max", max, "d=", num_pca)
