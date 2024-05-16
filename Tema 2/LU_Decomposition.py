import numpy as np

def factLU(A, tol=1.e-10):
    n = A.shape[0]
    mem = np.copy(A)
    M = np.zeros((n,n), float)
    for k in range(n):
        for i in range(k+1, n):
            if (abs(A[k, k]) < tol):
                raise ValueError("Diagonal element too small")
            else:
                M[i,k] = A[i,k]/A[k,k]
            for j in range(k, n):
                A[i,j] -= M[i,k]*A[k,j]
    for i in range(n):
        for j in range(i, n):
            M[i,j] = A[i, j]
    return M
