import numpy as np

#sinh ra 1000 gtri x
X = np.random.rand(1000, 1) 
y = 4 + 3*X
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

w = ((np.linalg.pinv((Xbar.T@(Xbar))))@Xbar.T)@y

def grad(w):   #đạo hàm hàm mất mát   #.dot = @ (dấu nhân)
    N = Xbar.shape[0]
    return 1/N * Xbar.T@(Xbar@w - y)
   #return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w): #hàm mất mát
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar@w, 2)**2;

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

w0 = np.array([[2], [1]])
(w1, it1) = myGD(w0, grad, 1)
print('Solution found by GD: w.T = ', w1[-1].T, ',after %d iterations.' %(it1+1))

#lấy file dl của bài ltrc mà đã làm, bài toán thị trường doanh thu, quảng cáo, dso. Làm 2 cách, 1 là = cthuc, 2 là làm bằng gradient