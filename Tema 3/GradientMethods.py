import numpy as np

def Gradient(A, b, x0, tol = 1.e-10, maxIter = 100000):
    r = b - (A @ x0)
    x = x0.copy()
    for k in range(maxIter):
        a = r.T @ r
        a /= r.T @ A @ r
        ar = a * r
        x += ar
        r -= A @ ar
        if (np.linalg.norm(r) <= tol):
            return x, k+1
    return None, -maxIter

def ConjugateGradient(A, b, x0, tol = 1.e-10, maxIter = 100):
    r = b - A @ x0
    p = r
    x = x0.copy()
    gamma = r.T @ r
    tol2 = tol ** 2
    for k in range(maxIter):
        y = A @ p
        alpha = gamma / (p.T @ y)
        x = x + alpha * p
        r = r - alpha * y
        r2 = r.T @ r
        beta = r2 / gamma
        gamma = r2 
        p = r + beta * p
        if (r.T @ r < tol2):
            return x, k+1
    return None, -maxIter

ç
N = 100
tol = 1.e-10
A = 2 * np.eye(N) + np.eye(N, k = 5) + np.eye(N, k = -5)
b = np.array(range(N))[:,np.newaxis] 
x0 = np.zeros((N,1))

_, k1= Gradient(A, b, x0)
_, k2 = ConjugateGradient(A, b, x0)
print("k1:", k1, "k2:", k2)
#Solució: Gradient: ~2167 iteracions, Gradient conjugat: 20.
