def metode_jacobi_vaps(A, tol = 1e-10 , maxRot = 1000):
    n = len(A)
    P = np.eye(n)
    
    for iter in range(maxRot):
        maxim = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if np.abs(A[i, j]) > maxim:
                    maxim = np.abs(A[i, j])
                    p, q = i, j
        if maxim < tol:
            return np.diag(A), P, iter

        eta = (A[p, p] - A[q, q]) / (2* A[p, q])
        t = - eta
        if eta > 0:
            t += - np.sqrt(eta**2 + 1)
            cos = 1 / np.sqrt(1 + t**2)
            sin = t*cos
        elif eta < 0:
            t += np.sqrt(eta**2 + 1)
            cos = 1 / np.sqrt(1 + t**2)
            sin = t*cos
        else:
            sin, cos = np.sqrt(2)/2, np.sqrt(2)/2
        
        R = np.eye(n)
        R[p, p] = R[q, q] = cos
        R[p, q], R[q, p] = -sin, sin
        A = R.T @ A @ R
        P = P @ R
    return np.diag(A), P, -iter
    


#################### PER TESTEJAR ####################

# Codi per testejar que funciona bé:
A = np.random.rand(10,10)
A = (A+A.T)/2 #Ara A és una matriu simètrica
vaps, veps = np.linalg.eig(A) # Retorna els vaps i una matriu amb els veps normalitzats
print("||Id - P^T@P|| =", np.linalg.norm(np.eye(len(veps)) - veps.T@veps))
print(np.sort(vaps))

vaps, veps, iter = metode_jacobi_vaps(A)
print("||Id - P^T@P|| =", np.linalg.norm(np.eye(len(veps)) - veps.T@veps))

print(np.sort(vaps))
print(f"Nombre de rotacions usades: {iter}")

# Funció de numpy per calcular els vaps i veps i perquè comproveu
