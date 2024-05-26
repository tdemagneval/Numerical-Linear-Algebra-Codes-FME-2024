import numpy as np

def iterPow (matA, z0, tol = 1.e-10, maxIter = 100):
    z = z0
    n = len(z0)
    sigma = 0
    for k in range(maxIter):
        z0 = z
        z = matA @z0
        z = z/np.linalg.norm(z)
        sigma0 = sigma
        sigma = z.T @ matA @ z
        if (abs(sigma - sigma0) < tol):
            return sigma, z, k+1
    return None, -maxIter

def dispPow(A, z0, q, tol=1.e-10, maxIter = 100):
    n = len(A)
    M = A - q * np.eye(n)
    vap, vep, iter = iterPow(M, z0, tol=tol, maxIter = maxIter) 
    return vap + q, vep, iter

def PA_LU_pivEsg (matA, tol = 1.e-10):
    A = matA.copy().astype(float)
    n = A.shape[0]
    P = np.array([i for i in range(n)])
    for k in range(n-1):
        i1 = k
        d = 0
        for i in range(k,n):
            s = max([abs(A[i,j]) for j in range(k,n)])
            if (abs(A[i,k])/s >= d):
                i1 = i
                d = abs(A[i,k])/s
        for j in range(n):
            s = A[i1,j]
            A[i1,j] = A[k,j]
            A[k,j] = s
        r = P[i1]
        P[i1] = P[k]
        P[k] = r
        for i in range(k+1, n):
            m = A[i,k]/A[k,k]
            A[i,k] = m
            for j in range(k+1,n):
                A[i,j] -= m*A[k,j]
    return np.eye(n)[P], A

def triL(L, b, tol=1e-10, ones = False):
    n = len(b)
    x = b
    for i in range(0, n):
        for j in range(0,i):
            x[i] -= L[i, j]*x[j]
        if (abs(L[i, i]) < tol):
            raise ValueError("Diagonal element too small")
        elif not ones:
            x[i] /= L[i,i]
    return x

def triU(U, b, tol=1e-10):
    n = len(b)
    x = b.copy()
    for i in range(n-1, -1,-1):
        for j in range(i+1,n):
            x[i] -= U[i, j]*x[j]
        if (abs(U[i, i]) < tol):
            raise ValueError("Diagonal element too small")
        else:
            x[i] /= U[i,i]
    return x

def PLU_solve (P, LU, b, tol = 1.e-10):
    b = P @ b
    y = triL(LU, b, ones = True)
    return triU(LU, y)

def iterInvPow (matA, z0, tol = 1.e-10, maxIter = 100):
    z = z0
    n = len(z0)
    P, LU = PA_LU_pivEsg(matA)
    sigma = 0
    for k in range(maxIter):
        z0 = z
        z = PLU_solve(P, LU, z)
        z = z/np.linalg.norm(z)
        sigma0 = sigma
        sigma = z.T @ matA @ z
        if (abs(sigma - sigma0) < tol):
            return sigma, z, k+1
    return None, -maxIter

def dispInvPow(matA, z0, q, tol=1.e-10, maxIter = 100):
    n = len(matA)
    M = matA - q * np.eye(n)
    vap, vep, iter = iterInvPow(M, z0, tol=tol, maxIter = maxIter)
    return vap+q, vep, iter
  
