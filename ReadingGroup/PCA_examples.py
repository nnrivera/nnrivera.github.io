import numpy as np
import plotnine as p9
import pandas as pd
import os
from scipy.stats import multivariate_normal

os.chdir("/home/nico/git/nnrivera.github.io/ReadingGroup")


###########Ejemplo 8.3 Libro HDS

nax = np.newaxis

np.random.seed(2023)
n=10000
dim = 6


alpha = 0.3
sigma = 1.5
theta = 0.01*np.arange(1,dim+1,1)

Id = np.eye(dim)
samples = np.random.multivariate_normal(np.zeros(dim), (sigma**2)*Id, size=n)

class_data = np.random.binomial(1, alpha, size=n)
class_data = (class_data-1/2)*2
data = theta[nax,:]*class_data[:,nax]+samples



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



def EM_algorithm(data, theta_0,sigma_0,alpha_0, iteraciones = 100):
    theta_iter = theta_0
    sigma2_iter = sigma_0**2
    alpha_iter = alpha_0
    
    for i in range(iteraciones):
        
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
    print("mu_EM=",theta_iter,"\nsigma_EM=",np.sqrt(sigma2_iter),"\nalpha_EM = ",alpha_iter)

    return theta_iter, np.sqrt(sigma2_iter), alpha_iter



##Implementación algoritmo EM
theta_0 = np.zeros(dim)
sigma2_0 = 2
alpha_0 = 0.8
theta_em, sigma_em, alpha_em = EM_algorithm(data, theta_0, sigma2_0, alpha_0, 100)

##Idea PAC


sigma_teo = np.outer(theta,theta)+(sigma**2)*np.eye(dim)
eigenvalues_teo, eigenvectors_teo = np.linalg.eig(sigma_teo)

Correlation_estimated = (np.transpose(data)@data)/n


eigenvalues, eigenvectors = np.linalg.eig(Correlation_estimated)
max_eigenvalue_index = np.argmax(eigenvalues)
max_eigenvalue = eigenvalues[max_eigenvalue_index]
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

eigenvalues=np.sort(eigenvalues)[::-1]
 
theta_est = np.sqrt(eigenvalues[0]-eigenvalues[1])*max_eigenvector
sigma_est = np.sqrt(eigenvalues[1])

print("theta estimado:", theta_est,"\nsigma estimado:", sigma_est)
