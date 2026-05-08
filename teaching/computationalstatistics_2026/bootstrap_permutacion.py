import numpy as np
import plotnine as p9
import pandas as pd

n=100
m = 50

X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,m)

Z = np.concatenate((X,Y))

df = pd.DataFrame(data = {"puntos":Z, "grupo":np.concatenate((np.zeros(n),np.ones(m)))})
df["grupo"] = df["grupo"].astype("category")

plot = p9.ggplot(data = df)+p9.aes(x = "puntos",y = 0, color = "grupo")+p9.geom_point(alpha = 0.3)
plot


def estadistico_malo(X,Y):
    return(np.sum(1*(X>1))-np.sum(1*(Y>1)),np.sum(1*(X>1))+np.sum(1*(Y>1)) )

def estadistico_bueno(X,Y):
    return(np.mean(X**2)-np.mean(Y**2))


def bootstrap(X,Y,M):
    Z = np.concatenate((X,Y))
    boostrap_statistic = np.zeros(M)
    for i in range(M):
        sigma = np.random.permutation(len(Z))
        Zsigma = Z[sigma]
        Xsigma = Zsigma[:len(X)]
        Ysigma = Zsigma[len(X):]
        boostrap_statistic[i] = estadistico_bueno(Xsigma,Ysigma)
    return boostrap_statistic



boostrap_samples = bootstrap(X,Y,10000)
df_bs = pd.DataFrame(data = {"valor_boostrap":boostrap_samples})

plot2 = p9.ggplot(data = df_bs)+p9.aes(x = "valor_boostrap")+p9.geom_density()


estadistico_bueno(X,Y)



