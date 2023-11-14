import numpy as np
import plotnine as p9
import pandas as pd
import os
from scipy.stats import multivariate_normal

#os.chdir("...") #change dir


###########Exampe 8.3 HDS



#np.random.seed(2023)

#Parameters of the model
n=5000
dim = 6
alpha = 0.3
sigma = 1.5
theta = np.arange(1,dim+1,1)

Id = np.eye(dim)
nax = np.newaxis

#generate data 
#The model is f(x) = \alpha N(x;\theta, \sigma^2 I) + (1-\alpha) N(x;-\theta,Sigma^2 I)$
#I think the book exchanges the role of $\theta$ and $-\theta$
samples = np.random.multivariate_normal(np.zeros(dim), (sigma**2)*Id, size=n)
class_data = np.random.binomial(1, alpha, size=n)
class_data = (class_data-1/2)*2
data = theta[nax,:]*class_data[:,nax]+samples


#plot data if dim =2
if dim ==2:
    df = pd.DataFrame(data)
    df.columns=["x", "y"]
    plot = (p9.ggplot(df)+p9.aes(x="x",y ="y")+p9.geom_point())
    print(plot)
    plot.save("PCA1.pdf")



##EM algorithm


def Likelihoods(data, theta_iter, sigma2_iter):
    
    multivariate_1 = multivariate_normal(mean=theta_iter, cov=sigma2_iter*Id)
    multivariate_0 = multivariate_normal(mean=-theta_iter, cov=sigma2_iter*Id)
    pdf_values_1 = multivariate_1.pdf(data)
    pdf_values_0 = multivariate_0.pdf(data)
    return pdf_values_1, pdf_values_0

def EM_algorithm(data, theta_0,sigma_0,alpha_0):
    theta_iter = theta_0
    sigma2_iter = sigma_0**2
    alpha_iter = alpha_0
    
    for i in range(10000):
        
        #compute likelihoods
        L1, L0 = Likelihoods(data, theta_iter, sigma2_iter)
        
        #Compute gammas
        gamma1 = alpha_iter*L1/(alpha_iter*L1+(1-alpha_iter)*L0)
        gamma0=1-gamma1
        
        #Compute new theta, sigma, alpha
        
        theta_new = np.sum((gamma1-gamma0)[:,nax]*data,axis=0)/n
        sigma2_new = (np.sum(gamma1*np.sum((data-theta_new[nax,:])**2,axis=1))+np.sum(gamma0*np.sum((data+theta_new[nax,:])**2,axis=1)))/(dim*n)
        alpha_new = np.sum(gamma1)/n
        
        theta_iter = theta_new
        sigma2_iter = sigma2_new
        alpha_iter = alpha_new
    print("mu=",theta_iter,"\nsigma=",np.sqrt(sigma2_iter),"\nalpha = ",alpha_iter)

    return theta_iter, np.sqrt(sigma2_iter), alpha_iter



##calling the EM algorithm
theta_0 = np.zeros(dim)
sigma2_0 = 2
alpha_0 = 0.8
#theta_em, sigma_em, alpha_em = EM_algorithm(data, theta_0, sigma2_0, alpha_0)

##Finding theta and sigma with PAC

Correlation_estimated = (np.transpose(data)@data)/n

eigenvalues, eigenvectors = np.linalg.eig(Correlation_estimated)
max_eigenvalue_index = np.argmax(eigenvalues)
max_eigenvalue = eigenvalues[max_eigenvalue_index]
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

eigenvalues=np.sort(eigenvalues)[::-1]
#According to our computations $lambda_0=\|\theta\|^2 + \sigma^2$ and $\lambda_1 = \sigma^2$
#Moreover, E(X) = (2\alpha-1)\theta$ so we can also find alpha 


#NOTE that the model has identifiability problems, thus, depending on the data we estimate \theta and alpha, or -\theta and (1-\alpha). It does not matter anyway
 
theta_est = np.sqrt(eigenvalues[0]-eigenvalues[1])*max_eigenvector
sigma_est = np.sqrt(eigenvalues[1])
alpha_est = np.mean((1/2)+(1/2)*(np.sum(data, axis =0)/n)/theta_est)


print(theta_est, sigma_est,alpha_est)
