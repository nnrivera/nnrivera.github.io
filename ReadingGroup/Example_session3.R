library(glmnet)
library(ggplot2)
library(reshape2)


#Some simulated data
set.seed(2023)
n = 1000
U = runif(n,0,4)
U = sort(U)
Y = 1/2*cos(U)-1/3*sin(3*U)+1/4*cos(17*U)+1/4+rnorm(n,0,1/10)
data = data.frame(U,Y) 

#Plot the data
grafico = ggplot(data = data)+aes(x=U,y=Y)+geom_point()
print(grafico)

#generate a lot of covariates with sin and cos's
d=25
X =c()
names = c("inter")
for(i in 1:d)
{
  X = c(X,cos(i*U), sin(i*U))
  names = c(names, paste("c",i,sep=""),paste("s",i,sep=""))
}

X = matrix(X, nrow = n, byrow= FALSE)


#four models: Linear Regression (LR), LR with L1 regularisation, LR with L2 regularisation, LR with both regularisations

model0 <- glmnet(X, Y, alpha = 0, lambda = 0)
model1 <- glmnet(X, Y, alpha = 1, lambda = 0)
model2 <- glmnet(X, Y, alpha = 0, lambda = 0.01)
model3 <- glmnet(X, Y, alpha = 1, lambda = 0.01)

#for plotting
df_LR = data.frame(names,as.vector(coef(model0)),as.vector(coef(model1)),as.vector(coef(model2)),as.vector(coef(model3)) )
colnames(df_LR)=c("fun", "0_LR", "1_LR", "2_LR-R2","3_LR-R1")
df_LR <- melt(df_LR, id.vars = "fun")
colnames(df_LR) = c( "fun", "A", "coeff")
grafico1 = ggplot(data = df_LR)+aes(x=fun,y=coeff)+geom_col()+ggtitle("LR")+facet_grid(vars(A))

#4th row shows that L1+L2 regularization is enough to kill most coefficient (they are actually 0)
#there is still a cos(4x) surviving, which is not part of the model. Beyond that, it works perfectly.
plot(grafico1)


#After this, we can adjust the model with the surviving coefficients
Xp =c(cos(U),sin(3*U),cos(17*U), cos(4*U))
Xp = matrix(Xp, nrow = n, byrow= FALSE)
model_new <- glmnet(Xp, Y, alpha = 0, lambda = 0)

#a very good fit :)
coef(model_new)



