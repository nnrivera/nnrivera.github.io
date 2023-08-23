rm(list = ls())
library("mvtnorm")

#Example 1.2.2 and 1.3.2 of HDS by Martin J. Wainwright.

#Number of data points and dimension
n = 10000
d = 200

#Data X ~ N(0, I_d)
X = rmvnorm(n, mean = rep(0, d), sigma = diag(d))

#estimated covariance matrix
Sigmahat = (t(X)%*%X)/n


#If d<<n, then eigenValues of Sigmahat are close to 1 (since Sigma = I_d)
#Otherwise, their distribution is close to the Marčenko–Pastur law
eigenValues <- eigen(Sigmahat)$values
hist(eigenValues)
summary(eigenValues)


####Solution: Hard-threshold as proposed in 1.3.2
#The idea works for any sparse covariance matrix, not only the identity

newSigmahat = Sigmahat*(abs(Sigmahat)>sqrt(2*log(d)/n))
new_eigenValues <- eigen(newSigmahat)$values
#they are very concentrated around 1, as it should for Sigma = Id
hist(new_eigenValues)
summary(new_eigenValues)
