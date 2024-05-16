def mgs_fact_QR(A):
    """
    Realiza la factorización QR de una matriz A utilizando el método de Gram-Schmidt modificado.
    
    Parámetros:
    A : np.array
        Matriz de dimensiones m x n para la cual realizar la factorización QR.
        
    Devuelve:
    A : np.array
        Matriz ortonormal obtenida al de dimensiones m x n.
    R : np.array
        Matriz triangular superior de dimensiones n x n.
    """
    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.zeros((m, n))
    V = np.zeros((m, n))
    for i in range(n):
        V[:,i] = A[:,i]
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:,i])
        Q[:,i] = V[:,i]/R[i,i]
        for j in range(i+1,n):
            R[i,j] = np.dot(Q[:,i],V[:,j])
            V[:,j] -= R[i,j]*Q[:,i]
    return Q, R

A_test = np.array([[1,0.0,1],[0,2,1],[1,-2,1],[0,1,0]], dtype=np.float64)
Q, R = mgs_fact_QR(A_test)
print(f"la solució és:\n Q: \n{Q}\n \nR: \n{R}\n")
print("QR-A = 0 masomenos: \n", Q@R-A)
