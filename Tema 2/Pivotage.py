# feu tot el codi (factorització amb pivotatge parcial esglaonat), i càlcul de matriu inversa vosaltres mateixos:
import numpy as np

def prettyP (v):
    n = len(v)
    P = np.zeros((n,n))
    for i in range(n):
        P[i, v[i]] = 1
    return P

def PA_LU_pivParcial (matA, tol = 1.e-10):
    A = matA.copy()
    n = A.shape[0]
    P = np.array([i for i in range(n)])
    for k in range(n-1):
        i1 = k
        s = abs(A[k,k])
        for i in range(k+1,n):
            if (abs(A[i,k]) > s):
                i1 = i
                s = abs(A[i,k])
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
    return prettyP(P), A

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
    return prettyP(P), A

def triL(L, b, tol=1e-10):
    n = len(b)
    x = b.copy()
    for i in range(0, n):
        for j in range(0,i):
            x[i] -= L[i, j]*x[j]
        if (abs(L[i, i]) < tol):
            raise ValueError("Diagonal element too small")
        else:
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

def invPivEsg(matA):
    n = matA.shape[0]
    P, LU = PA_LU_pivEsg(matA)
    U = np.triu(LU)
    L = LU - U + np.eye(n)
    # 1) resolver Ly = e_i para cada i
    # 2) resolver Ux_i = y
    # 3) cada c_i será una columna de I
    I = np.zeros((n,n))
    for i in range(n):
        y = triL(L, P[:,i])
        I[:,i] = triU(U, y)
    return I
    
print("Pivotatge parcial:")
A1_test = np.array([[0,1,2,3],[3,0,1,2],[2,3,0,1],[1,2,3,0]], dtype = float)
P1, LU1 = PA_LU_pivParcial(A1_test) 
print(f"P1 = \n {P1}\n \n LU1 = \n{LU1} \n")

U1 = np.triu(LU1)
L1 = LU1 - U1 + np.eye(4)
print("PA: \n", P1@A1_test, "\n")
print("LU: \n", L1@U1, "\n")


print("Pivotatge total:")
A2_test = np.array([[1,2,2,4],[1,2,1,2],[3,2,0,6],[4,2,1,1]])
P2, LU2 = PA_LU_pivEsg(A2_test)
print(f"P2 = \n {P2}\n \nLU2 = \n{LU2} \n")

U2 = np.triu(LU2)
L2 = LU2 - U + np.eye(4)
print("PA: \n", P2@A2_test, "\n")
print("LU: \n", L2@U2)

print("Inversa:")
B2 = invPivEsg(A2_test)
print(f"1/A = \n{B2} \n \nA * 1/A = identidad masomenos\n {A2_test@B2}")
