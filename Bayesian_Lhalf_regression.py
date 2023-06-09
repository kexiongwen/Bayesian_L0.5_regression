import numpy as np
from scipy import sparse
from scipy.stats import invgamma
from scipy.stats import invgauss

def Bayesian_L_half_regression(Y,X,M=10000,burn_in=10000):
    
    N,P=np.shape(X)
    w=1
    T1=1e-2

    #Initialization
    beta_sample=np.ones((P,M+burn_in))
    tau_sample=np.ones(P)
    v_sample=np.ones(P)
    a_sample=1
    sigma2_sample=np.ones(M+burn_in)
    lam_sample=1
    beta_sample[:,0:1]=np.random.randn(P,1)
        
    #MCMC loop
    
    for i in range(1,M+burn_in):
            
        #Sample beta
        sigma=(sigma2_sample[i-1]**0.5)
        D=tau_sample/lam_sample**2
        Mask1=D>T1
        D_diag=D[Mask1].reshape(-1,1)
        mu=np.random.randn(P,1)*D.reshape(P,1)
        DXT=D_diag*X[:,Mask1].T
        omega=DXT.T@DXT/sigma2_sample[i-1]+sparse.diags(np.ones(N))
        v=np.linalg.solve(omega, (Y/sigma-X@mu/sigma-np.random.randn(N,1)))
        beta_sample[:,i:i+1][Mask1]=mu[Mask1]+D_diag*DXT.dot(v)/sigma
        beta_sample[:,i:i+1][~Mask1]=mu[~Mask1]

        #Sample lambda
        lam_sample=np.random.gamma(2*P+0.5,((np.abs(beta_sample[:,i])**0.5).sum()+1/a_sample)**-1)
            
        #sample_a
        a_sample=invgamma.rvs(1)*(1+lam_sample)

        ink=lam_sample**2*np.abs(beta_sample[:,i])
            
        #Sample V
        v_sample=2/invgauss.rvs(np.reciprocal(np.sqrt(ink)))
        

        #Sample tau
        tau_sample=v_sample/np.sqrt(invgauss.rvs(v_sample/ink))

                
        #Sample sigma2
        rss=Y-X@beta_sample[:,i:i+1]
        sigma2_sample[i]=invgamma.rvs((w+N)/2)*0.5*(w+rss.T@rss)   
            
    #End of MCMC chain
             
    MCMC_chain=(beta_sample[:,burn_in:],sigma2_sample[burn_in:])
    
    return MCMC_chain
