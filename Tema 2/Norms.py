def pnorm (x, p):
    return sum(a**p for a in x) ** (1/p)

def norma1(v):    
    suma1 = sum(abs(x) for x in v)
    return suma1

def norma2(v):
    suma2 = np.sqrt(sum(x ** 2 for x in v))
    return suma2

def norma_inf(v):
    suma_inf = max(abs(x) for x in v)
    return suma_inf
