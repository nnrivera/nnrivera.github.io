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


### 5. Fabricando covariables extras y overfitting

#generamos estas covariables extra. Sabemos que no deberian ir en el modelo porque sabemos como se generaron los datos
#pero en la práctica no sabemos si deberian ir o no en el modelo, así que no está demás probar un poco
#la siguiente función agrega features polinomiales
def poly_features_2d(X, k):
    #agrega todos los polinomios entre grado 2 y k
    X1 = X[:, 1]
    X2 = X[:, 2]
    
    features = []
    
    for deg in range(2, k+1):  # grados 2,...,k
        for i in range(deg + 1):
            j = deg - i
            features.append((X1**i) * (X2**j))
    
    return np.column_stack([X] + features)

#acá conviene estandarizar ya que las nuevas columnas podrian tener valores muy grandes/chicos comparadas con las iniciales 
#ojo que el intercepto no se estandariza
X_aumentada = poly_features_2d(X,6)
X_aumentada_std = estandarizar_X(X_aumentada)

beta_aumentado_inicial = np.zeros(X_aumentada_std.shape[1])
beta_aumentado_estimado = gradient_descent(X_aumentada_std,y,beta_aumentado_inicial,100000,0.05)

#print(beta_aumentado_estimado) #podemos ver que le da harto valor a varios coeficientes de las variables inventadas. Algunos son cercanos a 0, pero otros no tanto 

#en la práctica, no conocemos el beta de verdad, así que es difícil saber que lo estamos haciendo mal, 
# sin embargo, podemos predecir en nuestros mismos datos y ver qué pasa
y_aumentado_pred = predecir(beta_aumentado_estimado, X_aumentada_std)


print("\n=== Accuracy ===")
print("Nosotros_original:  ", accuracy(y, y_pred_nosotros))
print("Statsmodels:        ", accuracy(y, y_pred_sm))
print("Sklearn:            ", accuracy(y, y_pred_sk))
print("Nosotros_polinomios:", accuracy(y, y_aumentado_pred)) #cuando tenemos overfitting el accuracy debería ser mayor, pero no siempre, ni tan exagerado


print("\n=== Loss function ===")
print("Nosotros_original:  ", loss_fun(X,beta_estimado,y))
print("Statsmodels:        ", loss_fun(X,beta_sm,y))
print("Sklearn:            ", loss_fun(X,beta_sk,y))
print("Nosotros_polinomios:", loss_fun(X_aumentada_std,beta_aumentado_estimado,y)) #la función de perdida debería ser menor

#Finalmente, veamos como se desempeñan los modelos cuando llegan nuevos datos
n_nuevo = 30 #treinta datos nuevos
X_nuevo = np.random.normal(0,1,size = (n_nuevo,d)) #generamos covariables
X_nuevo = np.hstack((np.ones((n_nuevo,1)),X_nuevo)) #intercepto
proba_nuevo = proba_fun(X_nuevo,beta_true) #encontramos las probabilidades reales asociadas a cada fila de X
y_nuevo = 1*(np.random.uniform(0,1,n_nuevo)<proba_nuevo) 

#En la vida real, solo recibimos X_nuevo, y debemos predecir y_nuevo. Obviamente acá, como conocemos la verdad, podemos usar
#la información extra para evaluar el desempeño

#primero, predecimos usando los betas de dimension 1+d
prediccion_nueva_original = predecir(beta_estimado, X_nuevo)

#para predecir en el modelo con extra covariables, debemos agregarlas a la matrix X_nuevo, y luego predecir
X_nuevo_poli =  poly_features_2d(X_nuevo,6)
X_nuevo_poli_std = estandarizar_X(X_nuevo_poli)
prediccion_nueva_polinomial = predecir(beta_aumentado_estimado, X_nuevo_poli_std)

print("\n=== Accuracy en Datos Nuevos ===") #el desempeño debería ser menor en general ya que no entrenamos con estos datos
print("Nosotros_original:  ", accuracy(y_nuevo, prediccion_nueva_original)) # esta debería ser parecida a la que obtuvimos cuando la probamos contra los mismos datos, quizás un poco peor
print("Nosotros_polinomios:", accuracy(y_nuevo, prediccion_nueva_polinomial)) #acá deberiamos empeorar más, debido a overfitting
