import numpy as np

def matHouseholder(u):
    n = len(u)
    u = np.reshape(u,(-1,1))
    H = np.eye(n)- (2/(u.T@u))*u@u.T
    return H

def iterPow (matA, z0, tol = 1.e-10, maxIter = 10000):
    z = z0
    n = len(z0)
    sigma = 0
    for k in range(maxIter):
        z0 = z
        z = matA@z0
        z = z/np.linalg.norm(z)
        sigma0 = sigma
        sigma = z.T @ matA @ z
        if (abs(sigma - sigma0) < tol):
            return sigma, z, k+1
    return None, None, -maxIter

def dispPow(A, z0, q, tol=1.e-10, maxIter = 100):
    n = len(A)
    M = A - q * np.eye(n)
    print(M)
    vap, vep, iter = iterPow(M, z0, tol=tol, maxIter = maxIter) 
    return vap + q, vep, iter

def deflation(A, tolPot = 1.e-12, maxIterPot = 500): 
    n = len(A)
    if n == 1:
        return A[0], np.array([[1.]])
    z0 = np.random.rand(n)
    vap, vep, iter = iterPow(A,z0,tol=tolPot,maxIter = maxIterPot)
    if vap is None: 
            vap, vep, iter = dispPow(A, z0, 1, tol=tolPot, maxIter = maxIterPot)
    u = vep.copy()
    if u[0] >= 0:
        u[0] += 1.
    else:
        u[0] -= 1.
    H = matHouseholder(u)
    Anew = H@A@H
    vaps, veps = deflation(Anew[1:,1:], tolPot = tolPot, maxIterPot = maxIterPot)
    vaps = np.append(vaps,vap)
    
    new_veps = np.zeros((n,n))
    new_veps[-1] = vep
    b = Anew[0,1:]
    for k, (vap2, vep2) in enumerate(zip(vaps, veps)):
        alpha = 0 if np.allclose(vap, vap2) else b.T @ vep2 / (vap2 - vap)
        new_veps[k] = np.append(alpha, vep2)
        new_veps[k] = H @ new_veps[k]
        new_veps[k] = new_veps[k] / np.linalg.norm(new_veps[k])

    return vaps, new_veps


# Matriu A per testejar:
n = 5
A = np.random.rand(5, 5)
A = (A+A.T)/2 # simètrica, per assegurar que isgui diagonalitzable


true_vaps, true_veps = np.linalg.eig(A)
true_veps = true_veps.T
# VAPs i VEPs
vaps, veps =deflacio(A)

# Retorno els veps ordenats:
args = np.argsort(true_vaps)
true_vaps = true_vaps[args]
true_veps = true_veps[args]

args = np.argsort(vaps)
vaps = vaps[args]
veps = veps[args]

print('VAPs: \n',true_vaps)
print('VAPs: \n',vaps)
print('VEPs: \n',true_veps)
print('VEPs: \n',veps)
