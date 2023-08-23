rm(list = ls())
#Example 1.2.3 of HDS by Martin J. Wainwright.

#number of data points
n = 1000

#we use monte-carlo method to compute the expected distance
#of a random point in $[0,1]^d$ to a set of $n$ random points in $[0,1]^d$

#We use different distances d, in the vector ds
ds = floor(seq(1,floor(sqrt(n)), length=20))
distances_ds =rep(0,length(ds))
M = 1000

for(j in 1:length(ds))
{
  d = ds[j]
  dist = rep(0,M)
  for (i in 1:M)
  {
    X = matrix(runif(n*d), ncol = d, nrow = n) #set of n random points in $[0,1]^d$
    Xp = runif(d) #random point in [0,1]^d
    dist[i] <- min(apply(abs(t(t(X)-Xp)), 1, max, na.rm=TRUE)) #\ell_{\infty} distance
  }
  distances_ds[j] = mean(dist)
}

plot(ds,distances_ds, xlab = "dimension", ylab = "expected distance")

           
