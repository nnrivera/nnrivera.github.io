rm(list = ls())
#Libraries used later for plotting
library(ggplot2)
library(reshape2)

## We create two functions
# the first generates samples from the median from independent data sets
# the second generates bootstrap samples from the median from 1 dataset


#n: sample size of the dataset
#m: number of datasets 
#rGen: sampler
RealDataMedian <- function(n,m,rGen)
{
  fun <-function(x){median(rGen(n))}
  samples = sapply(rep(0,m), FUN=fun )
  return(as.vector(samples))
}

#m = number of bootstraped datasets
#Data = datapoints in a vector
BootstrapMedian <- function(m, Data)
{
  fun = function(x){median(sample(Data, size = length(Data), replace = TRUE))}
  samples = sapply(rep(0,m), FUN = fun)
  return(as.vector(samples))
}

##Let's try our algorithm
#the first rGen does not fit in the theory
#the second rGen fits well
rGen = function(n){2*(rbinom(n,1,1/2)-1/2)*rbeta(n, 3,1)}
#rGen = function(n){rexp(n,1/4)}
n =10000

realMedian = sort(RealDataMedian(n,1000, rGen))
bootstrapMedian = sort(BootstrapMedian(4*n, rGen(n)))
estimatedNormalReal = sort(rnorm(1000, mean = mean(realMedian), sd = sd(realMedian)))
estimatedNormalBootstrap = sort(rnorm(1000, mean(bootstrapMedian), sd = sd(bootstrapMedian)))
#we have to plot a lot of stuff

AllData = c(realMedian,bootstrapMedian,estimatedNormalReal,estimatedNormalBootstrap)
AllCumulatives = c(1:length(realMedian)/length(realMedian),1:length(bootstrapMedian)/length(bootstrapMedian), 1:length(estimatedNormalReal)/length(estimatedNormalReal), c(1:length(estimatedNormalBootstrap))/length(estimatedNormalBootstrap))
Source = c(rep("real", length(realMedian)), rep("bootstrap", length(bootstrapMedian)), rep("est_normal_real", length(estimatedNormalReal)), rep("est_normal_bootstrap", length(estimatedNormalBootstrap)) )

df = data.frame(AllData, AllCumulatives, Source)
colnames(df) = c("median", "cumulative", "type")
plot_densities <- ggplot(df, aes(x = median, color = type))+
  geom_density(linewidth = 0.8)
plot_densities
ggsave("plot_density_bad_10000.pdf",plot_densities )

plot_cumulatives<- ggplot(df, aes(x = median, y = cumulative, color = type))+
  geom_line(linewidth=1.4)+
  geom_hline(yintercept = 0.025)+
  geom_hline(yintercept = 0.975)
  
plot_cumulatives
ggsave("plot_bad_10000.pdf",plot_cumulatives)
