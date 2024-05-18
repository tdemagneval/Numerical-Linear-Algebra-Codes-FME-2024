import numpy as np

def Jacobi (x0, matA, b, tol=1.e-10, maxIter=100):
    n = len(b)
    x = b.copy()
    for k in range(maxIter):
        x = b.copy()
        for i in range(n):
            for j in range(n):
                if (i != j): 
                    x[i] -= matA[i,j]*x0[j]
            x[i] /= matA[i,i]
        if (np.linalg.norm(x-x0) < tol * np.linalg.norm(x)):
            return x, k
        x0 = x
    return x, -maxIter


Jacobi(np.array([-10, -10, -10]), np.array([[2,1,0],[1,2,1],[2,0,2]]), np.array([3,4,4]))