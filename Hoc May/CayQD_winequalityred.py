import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
 
df = pandas.read_csv("winequality-red.csv", sep = ';')
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

X = df[features]
y = df['quality']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
# print(X)
print(dtree.predict([[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]]))


# from sklearn import tree
# fig, ax = plt.subplots(figsize=(10, 10))
# tree.plot_tree(dtree, feature_names=features, fontsize=10)
# plt.show()