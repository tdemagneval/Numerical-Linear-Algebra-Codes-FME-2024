import numpy as np

def Doolittle(A, tol=1.e-10): #returns M = L - Id + U
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n,n))
    for k in range(n):
        for j in range(k,n):
            U[k,j] = A[k,j]
            for r in range(k):
                U[k,j] -= L[k,r]*U[r,j]
        for i in range(k+1, n):
            L[i,k] = A[i,k]
            for r in range(k):
                L[i,k] -= L[i,r]*U[r,k]
            if (abs(U[k, k]) < tol):
                raise ValueError("Diagonal element too small")
            else:
                L[i,k] /= U[k,k]
    return L + U - np.eye(n)
