import numpy as np

def triU(U, b, tol=1e-10):
    n = len(b)
    x = b
    for i in range(n-1, -1,-1):
        for j in range(i+1,n):
            x[i] -= U[i, j]*x[j]
        if (abs(U[i, i]) < tol):
            raise ValueError("Diagonal element too small")
        else:
            x[i] /= U[i,i]
    return x
