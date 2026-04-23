import numpy as np

# 
def SM(A, B, mu):
    S = 0
    while S > -A and S < B:
        S += np.random.normal(mu, 1)
    return S

# 
def MonteCarlo(A, B, mu, N):
    total = 0
    for i in range(N):
        total = total+ (SM(A, B, mu) > B)
    return total / N

# 
def NuevoMonteCarlo(A, B, mu, N):
    total = 0
    auxiliar = np.zeros(N)
    for i in range(N):
        Sm_star = SM(A, B, -mu) #el nuevo estimador es simplemente cambiar la media $\mu$ por $-\mu$.
        auxiliar[i] = np.exp(2 * mu * Sm_star)
        total = total+ auxiliar[i] * (Sm_star > B)
    return total / N

# 
print(MonteCarlo(10, 10, -0.5, 100))
print(NuevoMonteCarlo(10, 10, -0.5, 100))

