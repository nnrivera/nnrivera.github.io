import numpy as np

n = 1000
beta = 0.2
iter = 10000

conf = np.zeros(n)
numero_de_caras = np.zeros(iter)

for i in range(iter):
    numero_de_caras[i]=np.sum(conf)
    listo = False
    entrada_a_cambiar = np.random.randint(0,n)
    confn = conf
    if conf[entrada_a_cambiar]==1:
        confn[entrada_a_cambiar]=0
    else:
        confn[entrada_a_cambiar]=1
        listo = True
    if not listo:
        #hay que ver si aceptamos o no la configuración
        if np.random.exponential(1)>beta*(2*np.sum(conf)-1):
            #se acepta
            listo = True
        else:
            listo = True
            confn = conf
    conf = confn #probablemente se puede optimizar un poco más para realizar menos oparaciones.




    
