import numpy as np
from scipy import sparse
from scipy.stats import invgamma
from scipy.stats import invgauss

def Bayesian_L_half_regression(Y,X,M=10000,burn_in=10000):
    
    N,P=np.shape(X)
    XTX=X.T@X
    XTY=X.T@Y
    YTY=Y.T@Y
    w=1

    T1=1e-2
    T2=1e-5
    T3=1e-5
    
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
        D_diag=sparse.diags(D[Mask1])
        mu=np.random.randn(P,1)*D.reshape(P,1)
        DXT=D_diag.dot(X[:,Mask1].T)
        omega=DXT.T@DXT/sigma2_sample[i-1]+sparse.diags(np.ones(N))
        v=np.linalg.solve(omega, (Y/sigma-X@mu/sigma-np.random.randn(N,1)))
        beta_sample[:,i:i+1][Mask1]=mu[Mask1]+D_diag.dot(DXT).dot(v)/sigma
        beta_sample[:,i:i+1][~Mask1]=mu[~Mask1]

        #Sample lambda
        lam_sample=np.random.gamma(2*P+0.5,((np.abs(beta_sample[:,i])**0.5).sum()+1/a_sample)**-1)
            
        #sample_a
        a_sample=invgamma.rvs(1)*(1+lam_sample)

        ink=lam_sample**2*np.abs(beta_sample[:,i])
            
        #Sample V
        Mask2=(ink<T2)
        v_sample[~Mask2]=2/invgauss.rvs(np.reciprocal(np.sqrt(ink[~Mask2])))
        v_sample[Mask2]=np.random.gamma(0.5,4*np.ones_like(v_sample[Mask2]))

        #Sample tau2
        Mask3=(ink<T3)
        tau_sample[~Mask3]=v_sample[~Mask3]/np.sqrt(invgauss.rvs(v_sample[~Mask3]/ink[~Mask3]))
        tau_sample[Mask3]=np.sqrt(np.random.gamma(0.5,2*np.square(v_sample[Mask3])))
                
        #Sample sigma2
        sigma2_sample[i]=invgamma.rvs((w+N)/2)*(0.5*w+0.5*YTY-beta_sample[:,i:i+1].T@XTY+0.5*beta_sample[:,i:i+1].T@XTX@beta_sample[:,i:i+1])   
            
    #End of MCMC chain
             
    MCMC_chain=(beta_sample[:,burn_in:],sigma2_sample[burn_in:])
    
    return MCMC_chain

def Conjugated_L_half(Y,X,M=10000,burn_in=10000):
    
    N,P=np.shape(X)
    w=1
    T1=1e-2
    T2=1e-5
    T3=1e-5
    
    #Initialization
    beta_sample=np.ones((P,M+burn_in))
    tau_sample=np.ones(P)
    v_sample=np.ones(P)
    a_sample=1
    sigma2_sample=np.ones(M+burn_in)
    lam_sample=1
    beta_sample[:,0:1]=np.random.randn(P,1)
    sigma=1
    
        
    #MCMC loop
    for i in range(1,M+burn_in):
            
        #Sample lambda
        lam_sample=np.random.gamma(2*P+0.5,((np.abs(beta_sample[:,i-1]/sigma)**0.5).sum()+1/a_sample)**-1)
            
        #sample_a
        a_sample=invgamma.rvs(1)*(1+lam_sample)

        ink=lam_sample**2*np.abs(beta_sample[:,i-1])/sigma
            
        #Sample V
        Mask2=(ink<T2)
        v_sample[~Mask2]=2/invgauss.rvs(np.reciprocal(np.sqrt(ink[~Mask2])))
        v_sample[Mask2]=np.random.gamma(0.5,4*np.ones_like(v_sample[Mask2]))

        #Sample tau2
        Mask3=(ink<T3)
        tau_sample[~Mask3]=v_sample[~Mask3]/np.sqrt(invgauss.rvs(v_sample[~Mask3]/ink[~Mask3]))
        tau_sample[Mask3]=np.sqrt(np.random.gamma(0.5,2*np.square(v_sample[Mask3])))

        #Sample sigma2
        D=tau_sample/lam_sample**2
        Mask1=D>T1
        D_diag=sparse.diags(D[Mask1])
        DXT=D_diag.dot(X[:,Mask1].T)
        omega=DXT.T@DXT+sparse.diags(np.ones(N))
        YT_omega_inv_Y=Y.T@np.linalg.solve(omega,Y)
        sigma2_sample[i]=invgamma.rvs((w+N)/2)*0.5*(w+YT_omega_inv_Y)

        #Sample beta
        sigma=(sigma2_sample[i]**0.5)
        mu=np.random.randn(P,1)*D.reshape(P,1)
        v=np.linalg.solve(omega, (Y/sigma-X@mu-np.random.randn(N,1)))
        beta_sample[:,i:i+1][Mask1]=sigma*(mu[Mask1]+D_diag.dot(DXT).dot(v))
        beta_sample[:,i:i+1][~Mask1]=sigma*mu[~Mask1]
                      
    #End of MCMC chain
                
    MCMC_chain=(beta_sample[:,burn_in:],sigma2_sample[burn_in:])
    
    return MCMC_chain