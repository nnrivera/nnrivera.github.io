##Este es un ejemplo donde Bootstrap falla!

rm(list = ls())
n = 264
Datos = rbeta(n, 0.5,2.3)+rexp(1,4)
#Generamos de la distribución del mínimo de verdad
N = 1000
minimos = rep(0,N)
for(i in 1:N)
{
  minimos[i]=min(rbeta(n, 0.5,2.3)+rexp(1,4))
}
hist(minimos)

##hacemos Bootstrap con M muestras usando nuestros Datos
M = 10000
minBootstrap = rep(0, M)
for( i in 1:M)
{
  dB = sample(Datos, n, replace = TRUE)
  minBootstrap[i] = min(dB)
}

#Graficamos la distribución acumulada empirica de Bootstrap
#y la comparamos con la de verdad
plot(sort(minBootstrap), (1:M)/M, xlim = c(min(minimos,minBootstrap), max(minimos,minBootstrap)))
lines(sort(minimos), (1:N)/N, col = "red")
#No se parecen en nada... 
#De hecho, la distriución de bootstrap está muy concentrada
plot(sort(minBootstrap), (1:M)/M)


