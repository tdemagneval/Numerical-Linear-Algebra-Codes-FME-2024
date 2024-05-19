import numpy as np

def pos_max(A, tol = 1.e-10):
    max = 0
    it_i,it_j = 0, 0
    n = A.shape[0]
    for i in range(n):
        for j in range(i):
            if i is not j and abs(A[i][j]) > max:
                max = abs(A[i][j])
                it_i = i
                it_j = j
    return it_i, it_j, max

def Jacobi(A, tol = 1.e-10, maxRot = 10000):
    n = len(A)
    iter = 0
    Q = np.eye(n)
    while iter < maxRot:
        p, q, max = pos_max(A)
        R = np.eye(n)
        m =(A[p][p]-A[q][q])/(2*A[p][q])
        if m > 0: 
            t = -m - np.sqrt(m**2 +1)
        elif m < 0:
            t = -m + np.sqrt(m**2 +1)
        else:
            t = 1
        cos = 1 /np.sqrt(1+t**2)
        sen = t*cos
        R[p][p], R[q][q] = cos, cos
        R[p][q] = -sen
        R[q][p] = sen
        
        A = R.T@A@R
        Q = Q @ R
        if max < tol:
            return np.diag(A), Q, iter
        iter = iter + 1
    return -iter
