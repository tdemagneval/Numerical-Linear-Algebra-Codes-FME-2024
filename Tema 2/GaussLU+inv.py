import numpy as np

def triL(matL, b, tol=1e-10, notOnes = True):
    L = matL.copy().astype(float)
    n = len(b)
    x = b.copy().astype(float)
    for i in range(0, n):
        for j in range(0,i):
            x[i] -= L[i, j]*x[j]
        if (abs(L[i, i]) < tol):
            raise ValueError("Diagonal element too small")
        elif notOnes:
            x[i] /= L[i,i]
    return x

def triU(matU, b, tol=1e-10):
    U = matU.copy().astype(float)
    n = len(b)
    x = b.copy().astype(float)
    for i in range(n-1, -1,-1):
        for j in range(i+1,n):
            x[i] -= U[i, j]*x[j]
        if (abs(U[i, i]) < tol):
            raise ValueError("Diagonal element too small")
        x[i] /= U[i,i]
    return x

def gaussLU (matA,  tol = 1e-10):
    A = matA.copy().astype(float)
    n = len(b)
    for k in range(n-1):
        for i in range(k+1, n):
            if (abs(A[k,k]) < tol):
                raise ValueError("Diagonal element too small")
                return None
            A[i, k] = A[i, k]/A[k, k]
            for j in range(k+1, n):
                A[i,j] -= A[i,k]*A[k,j]
    return A

def gaussSolve (matA, vectb , tol = 1e-10):
    A = gaussLU (matA, tol = tol)
    y = triL (A, vectb, notOnes = False, tol = tol )
    x = triU (A, y, tol = tol)
    return x

def LUinverse(A):
    M = gaussLU (A)
    n = M.shape[0]
    invA = np.zeros((n,n))

    for i in range(n):
        #Ly = ei
        y = triL(M, np.eye(n)[:,i], notOnes = False)
        #Ux=y
        x = triU(M, y)
        invA[:,i] = x
        #print(invA)
    return invA


x = gaussSolve (np.array([[1,2,3],[4,5,6],[7,8,10]]), np.array([10,11,12]))
print(x)

invA = LUinverse(A) 
print(invA)
print(np.linalg.inv(A))
