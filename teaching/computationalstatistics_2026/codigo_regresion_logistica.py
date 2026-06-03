import numpy as np

#Funciones auxiliares

def sigma_fun(z):
    return(1/(1+np.exp(-z)))

def proba_fun(X,beta):
    return sigma_fun(X@beta)

def estandarizar_X(X):
    X_std = X.copy()

    mu = X[:, 1:].mean(axis=0) #calcula medias por columnas, sin incluir la primera
    sigma = X[:, 1:].std(axis=0) #calculamos la desviación estandard por columns, sin incluir la primera
    sigma[sigma == 0] = 1

    X_std[:, 1:] = (X[:, 1:] - mu) / sigma
    return X_std


### 1. Generamos datos. En la vida real los datos los recibimos y no sabemos como fueron generados
###    pero acá los generamos para poder saber la verdad y así evaluar de mejor forma nuestro algoritmo
n = 400
d = 2 #número de covariables
X = np.random.normal(0,1,size = (n,d)) #generamos covariables
X = np.hstack((np.ones((n,1)),X)) #esto agrega el intercepto de forma limpia, no como en clases :)

beta_true = np.array([0.1,1,1]) #este es el verdadero beta que generará los datos
proba_nuevo = proba_fun(X,beta_true) #encontramos las probabilidades reales asociadas a cada fila de X

y = 1*(np.random.uniform(0,1,n)<proba_nuevo) #esto genera el vector y, usando las probabildades reales. 

### 2. Regresión Logística. Acá empezariamos en la vida real, observando el par X, y. Nuestro objetivo es estimar $beta$.

def gradiente(X, y,beta):
    
    n = len(y)
    grad = (1/n)* X.T @(proba_fun(X,beta)-y)
    return grad

def gradient_descent(X,y,beta_aumentado_inicial,iter,nu):
    beta = beta_aumentado_inicial

    for i in range(iter):
        beta = beta - nu*gradiente(X,y,beta)
    return(beta)

##Ahora ajustamos
beta_aumentado_inicial = np.zeros(1+d)
beta_estimado = gradient_descent(X,y,beta_aumentado_inicial,100000,0.05)

### 3. Otros modelos
import statsmodels.api as sm

model = sm.Logit(y, X)
result = model.fit(disp=0)
beta_sm = result.params


import sklearn.linear_model as lm



clf = lm.LogisticRegression(
    penalty='l2',
    C=np.inf,
    fit_intercept=False,
    max_iter=100000
)

clf.fit(X, y)
beta_sk = clf.coef_.flatten()


print("\n=== betas ===")
print("beta verdadero:", beta_true)
print("beta nosotros: ",beta_estimado)
print("beta sm:       ", beta_sm)
print("beta sklearn:  ", beta_sk)



### 4. Predecir y comparación



def predecir(beta, X, threshold=0.5):
    return 1*(proba_fun(X,beta) >= threshold)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def loss_fun(X, beta, y):
    p = proba_fun(X, beta)
    p = np.clip(p, 1e-30, 1 - 1e-30) #para evitar los problemas con p muy chicos, los truncamos. Probablemente igual no es la mejor solución para este problema numérico

    terms = -(y*np.log(p) + (1-y)*np.log(1-p)) #estos terminos son nan cuando el logaritmo se evalua en 0, pero **lo más probable** es que en esos casos el termino frente al logaritmo sea 0, además lim_{x\to 0}x\log(x) = 0.
    return np.nansum(terms) #suma ignorando nan

#como los betas son tan parecidos, essencialmente predecimos lo mismo.
y_pred_nosotros = predecir(beta_estimado, X)
y_pred_sm = predecir(beta_sm, X)
y_pred_sk = predecir(beta_sk, X)

print("\n=== Accuracy ===")
print("Nosotros:   ", accuracy(y, y_pred_nosotros))
print("Statsmodels:", accuracy(y, y_pred_sm))
print("Sklearn:    ", accuracy(y, y_pred_sk))
