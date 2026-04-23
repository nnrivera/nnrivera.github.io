from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import tree
import graphviz

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9

def plot_dots(X,y):
    df = pd.DataFrame({
    "x0": X[:, 0],
    "x1": X[:, 1],
    "y": y 
    })
    df["y"] = df["y"].astype("category") #transformamos en categórica
    plot = (p9.ggplot(data = df)+p9.aes(x = "x0", y = "x1", colour = "y", shape = "y")+p9.geom_point())
    return plot


def plot_region_tile(X,y,dec_tree, grid_size = 400):
    df = pd.DataFrame({
    "x0": X[:, 0],
    "x1": X[:, 1],
    "y": y  
    })
    df["y"] = df["y"].astype("category") #transformamos en categórica
    
    x0_grid = np.linspace(df.x0.min() - 0.5, df.x0.max() + 0.5, grid_size)
    x1_grid = np.linspace(df.x1.min() - 0.5, df.x1.max() + 0.5, grid_size)

    xx0, xx1 = np.meshgrid(x0_grid, x1_grid)

    #generamos la grilla y predecimos sobre los puntos de la grilla
    grid = np.c_[xx0.ravel(), xx1.ravel()]
    preds = dec_tree.predict(grid)

    grid_df = pd.DataFrame({
        "x0": grid[:, 0],
        "x1": grid[:, 1],
        "pred": preds
        })
    grid_df["pred"] = grid_df["pred"].astype("category")

    p = (
    p9.ggplot() #como cada geometría tiene sus propios datos, se los pasamos por separados, en vez de definirlos globalmente
    + p9.geom_tile(grid_df, p9.aes("x0", "x1", fill="pred"), alpha=0.3)
    + p9.geom_point(df, p9.aes("x0", "x1", color="y"), size=2)
    + p9.theme_minimal()
    )
    return p


#leemos datos
X, y = make_moons(n_samples=200, noise=0.25, random_state=1)

#plot datos originales
plot1 = plot_dots(X,y)
print(plot1)

##entrenamos un arbol de decisión. sklearn usa CART con opciones extra
##como el problema es de clasificación usamos el criterio de gini para separar
##Para encontrar el split optimo usamos "best" que encuentra la mejor combinacion de (covariable, separador)
##además se tiene max_depth que setea la profundidad máxima del arbol



#dec_tree = DTC(criterion="gini", splitter="best", random_state=1) #sin max depth podemos ver un overfit tremendo
dec_tree = DTC(criterion="gini", splitter="best", max_depth=5, random_state=1)
dec_tree.fit(X, y)

plt.figure(figsize=(10,6))
plot_tree = tree.plot_tree(dec_tree)
plt.show()

#vamos a predecir en casi todos los puntos
plot2 = plot_region_tile(X,y,dec_tree,grid_size = 400)

print(plot2)


#print(p)


### Perturbamos un poco los datos para ver que tan resistente es el árbol

X_perturbed = X.copy()
idx = np.random.choice(len(X), size=10, replace=False)
X_perturbed[idx] +=  np.random.normal(size=(len(idx), X.shape[1]))

#dec_tree = DTC(criterion="gini", splitter="best", random_state=1) #sin max depth podemos ver un overfit tremendo
dec_tree = DTC(criterion="gini", splitter="best", max_depth=3, random_state=1)
dec_tree.fit(X_perturbed, y)

#graficamos el árbol
plt.figure(figsize=(10,6))
plot_tree = tree.plot_tree(dec_tree)
plt.show()

#graficamos las regiones
plot3 = plot_region_tile(X_perturbed,y,dec_tree)
print(plot3)


###Finalmente un random forest se entrena igual que un árbol aleatorio, n_estimators es el número de árboles que se usarán

rf = RFC(n_estimators=100, criterion="gini", random_state=1, max_depth=5) #no tiene parametro splitter ya que siempre utiliza un split al azar, donde busca sobre un conjunto aleatorio de covariables




