import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from atexit import register
from cProfile import label
from tkinter import *
from tkinter import messagebox
from tokenize import Floatnumber
#precision: 
df = pd.read_csv('dataR2.csv')
X = np.array(df.loc[:, df.columns != "Classification"].values)
y = np.array(df["Classification"]).T

# splitting X and y into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# perceptron
pla = Perceptron()
main_pla = Perceptron()
# split the data
kf = KFold(n_splits=5)

maxScore = 0

# # generate indices to split data into training and test set.
for train_index, test_index in kf.split(X_train, y_train):
	new_X_train, new_X_test = X_train[train_index], X_train[test_index]
	new_y_train, new_y_test = y_train[train_index], y_train[test_index]
	pla.fit(new_X_train,np.ravel(new_y_train,order="C"))
	score_trained = pla.score(new_X_test, new_y_test)
	# print(score_trained)
	if score_trained > maxScore:
		maxScore = score_trained
		main_pla.fit(new_X_train, new_y_train)	
	

y_prediction = main_pla.predict(X_test)


#đánh giá perceptron
acc1 = accuracy_score(y_test,y_prediction)                    #so du doan dung/ toan bo cac du doan 
pre1 = precision_score(y_test,y_prediction, average = 'weighted')
rec1 = recall_score(y_test,y_prediction, average = 'weighted') #do luong ti le du doan chinh xac
f1_s1  = f1_score(y_test,y_prediction, average = 'weighted')

print("accuracy =",acc1, "precision =",pre1, " recall =",rec1, "\t" "F1 =", f1_s1 )

screen = Tk()
screen.geometry("500x450")
screen.title("Dự đoán khả năng bị ung thư vú ở người")

Age_text = Label(text="Age").place(x=15, y=30)
BMI_text = Label(text="BMI").place(x=15, y=80)
Glucose_text = Label(text="Glucose").place(x=15, y=130)
Insulin_text = Label(text="Insulin").place(x=15, y=180)
HOMA_text = Label(text="HOMA").place(x=280, y=30)
Leptin_text = Label(text="Leptin").place(x=280, y=80)
Adiponectin_text = Label(text="Adiponectin").place(x=280, y=130)
Resistin_text = Label(text="Resistin").place(x=280, y=180)
MCP_text = Label(text="MCP.1").place(x=280, y=230)

Age = DoubleVar()
BMI = DoubleVar()
Glucose = DoubleVar()
Insulin = DoubleVar()
HOMA = DoubleVar()
Leptin = DoubleVar()
Adiponectin = DoubleVar()
Resistin = DoubleVar()
MCP = DoubleVar()

Age_entry = Entry(textvariable=Age, width="30")
BMI_entry = Entry(textvariable=BMI, width="30")
Glucose_entry = Entry(textvariable=Glucose, width="30")
Insulin_entry = Entry(textvariable=Insulin, width="30")
HOMA_entry = Entry(textvariable=HOMA, width="30")
Leptin_entry = Entry(textvariable=Leptin, width="30")
Adiponectin_entry = Entry(textvariable=Adiponectin, width="30")
Resistin_entry = Entry(textvariable=Resistin, width="30")
MCP_entry = Entry(textvariable=MCP, width="30")

Age_entry.place(x=15, y=50)
BMI_entry.place(x=15, y=100)
Glucose_entry.place(x=15, y=150)
Insulin_entry.place(x=15, y=200)
HOMA_entry.place(x=280, y=50)
Leptin_entry.place(x=280, y=100)
Adiponectin_entry.place(x=280, y=150)
Resistin_entry.place(x=280, y=200)
MCP_entry.place(x=280, y=250)

def DuDoan():
	Age_info = Age.get()
	BMI_info = BMI.get()
	Glucose_info = Glucose.get()
	Insulin_info = Insulin.get()
	HOMA_info = HOMA.get()
	Leptin_info = Leptin.get()
	Adiponectin_info = Adiponectin.get()
	Resistin_info = Resistin.get()
	MCP_info = MCP.get()

	# tld_text = Label(text="Tỷ lệ dự đoán đúng : " + str(count_per/len(y_prediction)*100)+' %').place(x=15, y=360)
	# tls_text = Label(text="Tỷ lệ dự đoán sai : " +str(100 - count_per/len(y_prediction)*100)+' %').place(x=15, y=380)
	accuracy_text = Label(text="accuracy = " + str(acc1)).place(x=15, y=360)
	precision_text = Label(text="precision = "+str(pre1)).place(x=15, y=400)
	recall_text = Label(text="recall = "+str(rec1)).place(x=280, y=360)
	F1_text = Label(text="F1 = "+str(f1_s1)).place(x=280, y=400)

	sample_test = [[Age_info, BMI_info, Glucose_info, Insulin_info, HOMA_info, Leptin_info, Adiponectin_info, Resistin_info,MCP_info]]

	if main_pla.predict(sample_test) == 1:
		messagebox.showwarning('Kết quả dự đoán', 'Không bị ung thư vú!!')
	else:
		messagebox.showwarning('Kết quả dự đoán', 'Bị ung thư vú!!')

	Age_entry.delete(0, END)
	BMI_entry.delete(0, END)
	Glucose_entry.delete(0, END)
	Insulin_entry.delete(0, END)
	HOMA_entry.delete(0, END)
	Leptin_entry.delete(0, END)
	Adiponectin_entry.delete(0, END)
	Resistin_entry.delete(0, END)
	MCP_entry.delete(0, END)
register = Button(screen, text="Dự Đoán", width="35", height="2", command=DuDoan, bg='yellow')
register.place(x=150, y=300)
mainloop()
