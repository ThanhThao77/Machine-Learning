import numpy as np

A = np.array([[1, 2, 3],
              [2, 5, 3],
              [1, 0, 8]])
print("Ma trận A =\n", A)
print("Nhân ma trận với một số")
n = int(input("Nhập số n = "))
print(A * n)

# Chuyen vi ma tran
print("\nMa trận chuyển vị A^T =\n", A.T)

# Cong, tru hai ma tran
B = np.array([[2, 3, 4],
              [-1, -2, -3],
              [0, 4, -4]])
print("\nCộng 2 ma trận: A + B =\n", A + B)
print("\nTrừ 2 ma trận: A - B =\n", A - B)

# Nhan hai ma tran: số cột của ma trận 1 = số dòng ma trận 2
print("\nNhân 2 ma trận:")
print("A * B =\n", A @ B)
print("B * A =\n", B @ A)

# Tim ma tran nghich dao
print("\nMa trận nghịch đảo: A^(-1) =\n", np.linalg.inv(A))

#kiem tra xem có phải ma trận nghịch đảo không