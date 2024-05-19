import numpy as np

'''
Input:
 - La matriu A de la cual volem l'espectre
 - Tolerància (parem quan tots els valors de fora la diagonal són menors que tol en abs)
 - maxRot (autodescriptiu)
'''

def jacobi(A, tol, maxRot):
    n = (A.shape)[0]
    Q = np.identity(n)
    
    for rot_number in range(maxRot):
        p = 0
        q = 1
        Apq = abs(A[p][q])
        for i in range(0, n):
            for j in range(i+1, n):
                if abs(A[i][j]) > Apq:
                    Apq = abs(A[i][j])
                    p,q = i,j
        if Apq < tol:
            VAPs = [A[i][i] for i in range(0,n)]
            VEPs = Q
            return [VAPs, VEPs, rot_number]
        
        eta = (A[p][p]-A[q][q])/(2*A[p][q])
        t = -eta-np.sqrt(1 + eta**2)
        phi = np.arctan(t)
        
        Qi = np.identity(n)
        Qi[p][p] = np.cos(phi)
        Qi[q][q] = np.cos(phi)
        Qi[p][q] = -np.sin(phi)
        Qi[q][p] = np.sin(phi)

        A = np.dot(np.dot(np.transpose(Qi), A), Qi)
        Q = np.dot(Q, Qi)

    return [-maxRot]

#################### PER TESTEJAR ####################

# Codi per testejar que funciona bé:
A = np.random.rand(5,5)
A = (A+A.T)/2 #Ara A és una matriu simètrica

# Funció de numpy per calcular els vaps i veps i perquè comproveu que teniu els mateixos resultats
vaps, veps = np.linalg.eig(A) # Retorna els vaps i una matriu amb els veps normalitzats

print("TESTEIG PER CONTRASTAR")
print(vaps)
print(veps)

print("\nRESULTATS DE LA FUNCIÓ JACOBI")

# Crideu la vostra funció:

l = jacobi(A, 1.e-5, 100)

if len(l) == 1:
    print(-l[0])
else:
    print("VAPs:")
    print(l[0])
    print("\nVEPs (la columna i d'aquesta matriu és VEP del i-èssim element de l'array de VAPs):")
    print(l[1])
    print("\nNombre de rotacions usades")
    print(l[2])
