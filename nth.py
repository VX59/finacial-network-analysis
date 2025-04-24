import numpy as np
A = np.array([[0,1,1,0],
              [1,0,1,0],
              [1,1,0,1],
              [0,0,1,0]])

one = np.ones(len(A))

k = np.matmul(A,one)
m = np.dot(k,one)/2
print(k)

N = np.matmul(A,A.transpose())

print(N)