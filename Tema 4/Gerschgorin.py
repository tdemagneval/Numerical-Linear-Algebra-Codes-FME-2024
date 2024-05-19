import numpy as np
import matplotlib.pyplot as plt

def discsGerschgorin(A):
    n, m = A.shape
    if n != m:
        raise Exception("Error! La matriu no és quadrada")
    c = np.diag(A)
    r = np.linalg.norm(A, axis = 1, ord = 1) - np.absolute(c)
    return c, r

def plotDiscs(A, xmin = -20, xmax = 20):
    n = A.shape[0]
    c,r = discsGerschgorin(A)
    c2,r2 = discsGerschgorin(A.T)
    vaps, veps = np.linalg.eig(A)
    fig, ax = plt.subplots()
    ax.set_aspect('equal') 
    ax.grid(True)
    for i in range(n):
        circle = plt.Circle((np.real(c[i]), np.imag(c[i])), r[i], color='blue', fill=True, alpha=0.3)
        ax.add_patch(circle)
        circle = plt.Circle((np.real(c2[i]), np.imag(c2[i])), r2[i], color='orange', fill=True, alpha=0.3)
        ax.add_patch(circle)
        # Limits dels eixos:
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.scatter(np.real(vaps),np.imag(vaps),color="black", s=5)
    plt.xlabel('Parte real')
    plt.ylabel('Parte imaginária')
    plt.show()
    return None
   
A = np.array([[7,-1,1,1,-1],[1,-5,0,0,-1],[0,0,14,-1,1],[-1,-1,0,-12,1],[0,1,1,0,0]],dtype = np.float64)
plotDiscs(A)
