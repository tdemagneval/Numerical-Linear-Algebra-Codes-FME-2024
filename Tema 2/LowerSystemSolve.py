import numpy as np

def triL(L, b, tol=1e-10):
    n = len(b)
    x = b
    for i in range(0, n):
        for j in range(0,i):
            x[i] -= L[i, j]*x[j]
        if (abs(L[i, i]) < tol):
            raise ValueError("Diagonal element too small!")
        else:
            x[i] /= L[i,i]
    return x
