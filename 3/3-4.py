import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A.shape)
print(B.shape)
print(np.dot(A, B))

C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[1, 2], [3, 4], [5, 6]])

print(C.shape)
print(D.shape)
print(np.dot(C, D))
