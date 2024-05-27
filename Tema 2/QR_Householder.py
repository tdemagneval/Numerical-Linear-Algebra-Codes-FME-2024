import numpy as np

def norm2(v):    
    return np.sqrt(sum(x ** 2 for x in v))

def QR_Householder (matA):
    A = matA.copy().astype(float)
    m, n = A.shape
    Q = np.eye(m)
    for k in range(n):
        x = np.array(A[k:m, k])
        v = x[:,np.newaxis]
        v[0] = v[0] + norm2(v) if v[0] >= 0 else v[0] - norm2(v)
        v /= norm2(v)
        A[k:m, k:n] -= v @ (2 * (v.T @ A[k:m, k:n]))
        H_k = np.eye(m-k) - v @ (2 * v.T)
        Q[:, k:] = Q[:, k:] @ H_k
    return np.round(Q[:, :n], 8), np.round(A[:n,:], 8)

def QR_Householder_Qtb (matA, b):
    '''importante, el householder Qtb no resuelve min(Ax-b) sino que devuelve R y Qtb tal que la solución de minimos 
    cuadrados es la solución de Rx = Qtb con R triangular superior, hay que usar triU(R, Qtb) para resolver el sistema'''
    R = matA.copy().astype(float)
    Qtb = b.copy().astype(float)[:,np.newaxis]
    m, n = R.shape
    for k in range(n):
        x = np.array(R[k:m, k])
        v = x[:,np.newaxis]
        v[0] = v[0] + norm2(v) if v[0] >= 0 else v[0] - norm2(v)
        v /= norm2(v)
        R[k:m, k:n] -= v @ (2 * (v.T @ R[k:m, k:n]))
        Qtb[k:m] -= v @ (2 * (v.T @ Qtb[k:m]))
    return np.round(Qtb[:n].T[0], 8), np.round(R[:n,:], 8)


print("test Q y R")
A_test = np.array([[1,0,1],[0,0,1],[1,-2,1],[0,1,0]])
Q_test, R_test = QR_Householder (A_test)
Q, R = np.linalg.qr(A_test)
print(Q_test,'\n \n', np.round(Q, 8), '\n')
print(R_test,'\n \n', np.round(R, 8))

print("test Qtb y R")

A_test = np.array([[1,0,1],[0,0,1],[1,-2,1],[0,1,0]])
b_test = np.array([1, 2, 1, 3])
Qtb_test, R_test = QR_Householder_Qtb (A_test, b_test)
Q, R = np.linalg.qr(A_test)
Qtb = Q.T @ b_test
print(Qtb_test,'\n\n', np.round(Qtb, 8), '\n')
print(R_test,'\n\n', np.round(R, 8))
