import os
import pandas as pd
import statsmodels as sm
import sklearn as sk
import numpy as np

df = pd.read_csv("/home/nico/aux_folder/iris/iris.data")
df["tipo"] = 1*(df["tipo"]=="Iris-setosa")


X = df[["x1", "x2", "x3", "x4"]].to_numpy()
y = df["tipo"].to_numpy()

X = (X-X.mean(axis=0))/np.sqrt(X.var(axis=0))


n_iter = 1000
n = X.shape[0]
d = X.shape[1]
m = 3 #tamaño de la capa oculta
nu = 0.1 #tamaño del salto

W = np.random.normal(loc = 0, scale = 1,size= (m,d))
b = np.random.normal(loc = 0, scale = 1, size = m)
a = np.random.normal(loc = 0, scale = 1, size = m)
c = np.random.normal(loc = 0, scale = 1, size = 1)




def sigma(x):
    return(1/(1+np.exp(-x)))

def logloss(s,y):
    return -np.mean(y*np.log(s+ 1e-8)+(1-y)*np.log(1-s+1e-8))

for i in range(n_iter):
    ##Forward

    Z = sigma(W@X.T+b[:, None]).T #tenemos todos los Z para todos los datos. Z_k = Z[k,:]
    S = sigma(Z@a+c)#f_theta(X)
    print(logloss(S, y))
    #Backward

    Dc = np.mean((S-y))
    Da = np.mean((S-y)[:, None]*Z, axis = 0)
    Db = np.mean((S-y)[:,None]*(a[None,:]*(Z*(1-Z))), axis = 0)
    DW = ((S-y)[:,None]*(a[None,:]*(Z*(1-Z)))).T@X/n
    
    c = c - nu*Dc
    a = a - nu*Da
    b = b - nu*Db
    W = W - nu*DW

    


print(logloss(S, y))



resultados = pd.DataFrame({"S":S, "y": y})
