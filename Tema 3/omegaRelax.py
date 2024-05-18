import numpy as np

def iterRel (w, x0, matA, b, tol=1.e-10, maxIter=100):
    n = len(b)
    x = x0.copy()
    for k in range(maxIter):
        for i in range(n):
            x[i] = b[i]
            for j in range(n):
                if (j < i): 
                    x[i] -= matA[i,j]*x0[j]
                if (j > i):
                    x[i] -= matA[i,j]*x[j]      
            x[i] *= w/matA[i,i]
            x[i] += (1 - w)*x0[i]
        if (np.linalg.norm(x-x0) < tol*np.linalg.norm(x)):
            return x, k
        x0 = x.copy()
    return x, -maxIter
    
    

for w in range(1, 20):
    print(w/10, iterRel (w/10, np.array([1, 0.5]), np.array([[2,1],[1,3]]), np.array([1,0])))
