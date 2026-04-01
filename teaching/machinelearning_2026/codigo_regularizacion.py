import pandas as pd
import os 
import numpy as np


#Leemos los datos
#os.chdir("...") probablemente tengan que cambiar el directorio al directorio donde tienen los datos
data = pd.read_csv("data_regresion.csv")

#Extraemos la respuesta y, y las covariables respectivas
y = data["target"].values
X = data.drop(columns = ["target"]).values

# tamaño de muestra y dimension
n,d = X.shape

#estandarizamos
means = X.mean(axis = 0)
sigma = X.std(axis = 0)

for i in range(d):
    X[:,i]= (X[:,i]-means[i])/sigma[i]

#de acá en adelante X está estandarizada

ones = np.ones((n,1))
X = np.hstack((ones, X))

#vamos a hacer regresion, penalizando a todos menos a beta_0

def regresion_ridge(X,y,lam = 0):
    n,d = X.shape 
    I = np.eye(d)
    I[0,0]=0 #deja de penalizar a beta_0
    beta = np.linalg.inv(((X.T@X)+n*lam*I))@(X.T@y)
    return beta



def soft_threshold(a,lam):
    return np.sign(a)*np.max((np.abs(a)-lam,0))

def regresion_lasso(X,y,lam=0, iter=1000):
    n, d = X.shape
    beta = np.zeros(d)
    
    #primero el vector b 
    b = (2/n)*np.sum((X*X),axis = 0)
    for i in range(iter):
        for j in range(d):
            #calcular el a_j
            rj = y-X@beta+beta[j]*X[:,j]#r_{-j} Este cálculo se puede optimizar, el calcular X@beta es muy costoso, siendo que el vector beta a cambiado muy poco
            aj = -2/n*np.dot(X[:,j],rj)# en cases se me olvidó multiplicar por -2/n. Que no les pase a ustedes :(
            #print(rj,aj)

            if j==0: #aca no hay que penalizar
                beta[j] = soft_threshold(-aj,lam)/b[j]
            else:
                beta[j] = soft_threshold(-aj,lam)/b[j]
    return beta


beta_ridge0 = regresion_ridge(X,y,lam=0)
beta_ridge1 = regresion_ridge(X,y,lam=0.1)
beta_ridge2 = regresion_ridge(X,y,lam=0.5)
beta_ridge3 = regresion_ridge(X,y,lam=0.1)
beta_ridge4 = regresion_ridge(X,y,lam=10)

beta_lasso0 = regresion_lasso(X,y,0, 10000)
beta_lasso1 = regresion_lasso(X,y,0.1, 10000)
beta_lasso2 = regresion_lasso(X,y,0.5, 10000)
beta_lasso3 = regresion_lasso(X,y,1, 10000)
beta_lasso4 = regresion_lasso(X,y,10, 10000)


#primera comparación de ambos métodos sin regularizar (i.e \lambda = 0)

#podemos ver que son casi iguales, excepto en esos coeficientes sospechosamente grandes de ridge
#ojo que ambos métodos son diferentes acá: uno utiliza la formula directa, el de lasso es iterativo,
#y a pesar de resolver el mismo problema de optimización las soluciones obtenidas son distintas

print(beta_ridge0)
print(beta_lasso0)

#Un poco de regularización. Lasso elimina inmediatemente algunos coeficientes quedandonos con solo 5
#Ridge elimina los valores gigantes que aparecieron, pero los valores chicos están ahí.
#Noten que los valores distintos de 0 de lasso igual coinciden con valores bien chiquititos de la regresión ridge
#sin embargo, en donde hay valores grandes de lasso, algunos se ven un poco distintos que los de ridge
print(beta_ridge1)
print(beta_lasso1)

#harta regularización. Lasso tiene sospechosamente solo 2 covariables vivas, mientras que ridge tiene entre 3 y 5 covariables con valores altos
print(beta_ridge4)
print(beta_lasso4)

#Finalmente, si quisieramos probar el desempeño de nuestros algoritmos deberiamos separar en set de test y entrenamiento
