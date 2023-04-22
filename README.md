# Bayesian $L_{1/2}$ regression

Here is the Python and Matlab implementation of High dimensional-Bayesian linear regression with $L_{1/2}$ prior based on our paper https://arxiv.org/pdf/2108.03464.pdf.



## Model setting

Consider a Gaussian linear model with likelihood


$$
L\left(Y \mid X \beta, \sigma^{2}\right)=\left(2 \pi \sigma^2\right)^{-N / 2} \exp\left[{-\frac{1}{2 \sigma^2}(Y-X \beta)^{T}(Y-X \beta)}\right]
$$

where $X$ is a $N \times P$ matrix of covariates, $\beta \in \mathbb{R}^{P}$ is assumed to be a sparse vector, and $Y \in \mathbb{R}^N$ is an $N$-vector of response observations. We assign a $L_{\frac{1}{2}}$ prior to $\beta$, such that


$$
\pi(\beta_{j}\mid \lambda) \propto \exp[-\lambda|\beta_{j}|^{\frac{1}{2}}]
$$


with hyper-prior for $\lambda$:


$$
\frac{1}{\sqrt{\lambda}} \sim \mathrm{Cauchy}_{+}(0,1)
$$


We show that the above prior setting has the following Global-local shrinkage hierarchical structure:


$$
\begin{gathered}
\beta_{j} \mid  \lambda, \tau_{j}^{2}  \stackrel{i i d}{\sim} \mathrm{N}\left(0,  \frac{\tau_{j}^{2}}{\lambda^{4}}\right) \quad \text{for} \quad j=1, \ldots, p \\
\tau_{j}^{2}\mid v_{j} \stackrel{i i d}{\sim} \mathrm{Exp}\left(\frac{1}{2v_{j}}\right) \quad \text{and} \quad v_{j}\stackrel{i i d}\sim \mathrm{Gamma}\left(\frac{3}{2},\frac{1}{4}\right) \quad \text{for} \quad j=1, \ldots, p\\
\lambda \sim \operatorname{Gamma}\left(\frac{1}{2}, \frac{1}{b}\right), \quad b \sim \operatorname{InvGamma}\left(\frac{1}{2}, 1\right)
\end{gathered}
$$



## Partially Collapsed Gibbs Sampler

We develop the PCG sampler that targets the exact horseshoe posterior with the following update rule:



S1. Sample $\beta \sim \mathrm{N}_{P}\left(\left(X^{T} X+\sigma^2 D^{-1}\right)^{-1} X^{T} Y, \sigma^{2}\left(X^{T} X+\sigma^{2} D^{-1}\right)^{-1}\right)$



S2. Sample $\lambda \sim$ Gamma $\left(2P+0.5, \sum_{j=1}^{P} |\beta_{j}|^{1/2}+1/b \right)$



S3. Sample $\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right), \quad j=1, \ldots, p$ 



S4. Sample $\frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}^{2}}\right), \quad j=1, \ldots, p$



S5. Sample $\sigma^{2} \sim \operatorname{InvGamma}\left(\frac{N}{2}, \frac{1}{2}\left(Y-X \beta\right)^T\left(Y-X \beta\right)\right)$



S6. Sample $b \sim \operatorname{InvGamma~}\left(1,1+\lambda\right)$



Where $D= \frac{1}{\lambda^{4}}\mathrm{Diag}(\tau^{2})$.



**Note that unlike the traditional Gibbs sampler, some of the order for the update does matter in PCG sampler!!!**



## Usage

```
beta_sample,sigma2_sample=Bayesian_L_half_regression(Y,X,M=10000,burn_in=10000) 

beta_mean=np.mean(beta_sample,axis=1)

beta_median=np.median(beta_sample,axis=1)

sigma2_mean=np.mean(sigma2_sample)

sigma2_median=np.median(sigma2_sample)
```




$Y$ is the vector of response with length $N$ and $X$ is $N \times P$ covariate matrix. $M$ is the number of the samples from MCMC with default setting 10000. burn_in is the burn in period for MCMC with default setting 10000.  



## Reference



```
@article{ke2021bayesian,
  title={Bayesian $ L_\frac{1}{2}$ regression},
  author={Ke, Xiongwen and Fan, Yanan},
  journal={arXiv preprint arXiv:2108.03464},
  year={2021}
}
```



