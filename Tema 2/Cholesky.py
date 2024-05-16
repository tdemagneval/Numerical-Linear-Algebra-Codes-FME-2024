'''
Genereu el codi, utilitzant numpy per resoldre, utilitzant el mínim nombre d'operacions possibles el càlcul de la inversa. ës a dir, haureu de fer:

1) La implementació del càlcul de la matriu L
2) La implementació de la resolució del sistemes triangulars.
3) El càlcul de la inversa

'''
import numpy as np
from math import sqrt

def cholesky(matA, tol = 1.e-10):
    L = matA.copy().astype(float)
    n = L.shape[0]
    for k in range(n):
        for r in range(k):
            L[k,k] -= L[k,r]**2
        if(L[k,k] < tol): 
            raise ValueError("Diagonal element too small")
            return None
        L[k,k] = sqrt(L[k,k])
        for i in range(k+1,n):
            for r in range(k):
                L[i,k] -= L[i,r]*L[k,r]
            L[i,k] /= L[k,k]
    return np.tril(L) #eliminamos los L[i,j] dónde i < j

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

def choleskyInv(matA):
    L = cholesky(matA)
    # 1) resolver Ly = e_i para cada i
    # 2) resolver L.Tx_i = y
    # 3) cada c_i será una columna de I
    n = matA.shape[0]
    I = np.zeros((n,n))
    for i in range(n):
        y = triL(L, np.eye(n)[i])
        I[:,i] = triU(L.T, y)
    return I

print(np.eye(3)[:,0])
A_test = np.array([[13, 11, 11],[11,13,11],[11,11,13]])
L = cholesky(A_test)
print(f"L: \n {L} \n")
L2 = np.linalg.cholesky(A_test)
print(f"L de numpy: \n {L2} \n")

B = choleskyInv(A_test)
print(f"1/A: \n {B} \n")

print("A * 1/A = Id masomenos")
print(A_test@B)
