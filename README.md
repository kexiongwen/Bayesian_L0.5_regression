# Bayesian $L_{1/2}$ regression

Here is the python implementation of High dimensional-Bayesian linear regression with $L_{1/2}$ prior based on our paper https://arxiv.org/pdf/2108.03464.pdf.



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

We first define some quantities that will be used in the PCG sampler scheme:


$$
D= \frac{1}{\lambda^{4}}\mathrm{Diag}(\tau^{2}), \quad M= \frac{XDX^{T}}{\sigma^{2}}+I_{N}
$$



We develop the PCG sampler that targets the exact horseshoe posterior with the following update rule:



S1. Sample $\beta \sim \mathrm{N}_{P}\left(\left(X^{T} X+\sigma^2 D^{-1}\right)^{-1} X^{T} Y, \sigma^{2}\left(X^{T} X+\sigma^{2} D^{-1}\right)^{-1}\right)$



S2. Sample $\lambda \sim$ Gamma $\left(2P+0.5, \sum_{j=1}^{P} |\beta_{j}|^{1/2}+1/b \right)$



S3. Sample $\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right), \quad j=1, \ldots, p$ 



S4. Sample $\frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right), \quad j=1, \ldots, p$



S5. Sample $\sigma^{2} \sim \operatorname{InvGamma}\left(\frac{N}{2}, \frac{1}{2}\left(Y-X \beta\right)^T\left(Y-X \beta\right)\right)$



S6. Sample $b \sim \operatorname{InvGamma~}\left(1,1+\lambda\right)$



**Note that unlike the traditional Gibbs sampler, some of the order for the update does matter in PCG sampler!!!**



The efficient method from https://arxiv.org/pdf/1506.04778.pdf has been used to sample from the conditional posterior $\pi(\beta \mid \sigma^{2}, \lambda, \tau^2)$:




1. Sample $u  \sim \mathrm{N}(0, D)$ and $f \sim \mathrm{N}\left(0, I_{N}\right)$

3. Set   $v =\frac{Xu}{\sigma}+f$ and $v^{*}=M^{-1}(Y / \sigma-v)$

4. Set  $\beta=u+\frac{1}{\sigma}DX^{T}v^{*}$

   

## JOB-approximation

To further  the reduce computational cost per step, we employ the strategy from  https://jmlr.org/papers/v21/19-536.html  to approximate the matrix product $XDX^{T}$ by hard-thresholding , resulting in


$$
M \approx I_{N}+\frac{1}{\sigma^{2}}XD_{\delta}X, \quad D_{\delta}=\frac{1}{\lambda^{4}}\mathrm{Diag}(\tau_{j}^{2}1({\tau_{j}^{2}/\lambda^{4}>\delta}))
$$


for “small” $\delta$. Using this strategy,the approximate algorithm uses the same update rule as in the MCMC scheme above, with only two changes:



1. $M$ is replaced by $M_{\delta}$ everywhere that it appears in the PCG sampler; and

2. In the final step of sampling $\pi(\beta \mid \sigma^{2}, \lambda, \tau^2)$, the quantity $DX^{T}$ is replaced by $D_\delta X^{T}$



Finally, we set the hard-threshold  $\delta=1e-4$ as suggested by their paper.



## Extra approximation to improve the numerical stability 

In PCG sampler scheme for $L_{1/2}$ prior, we need to sample 



$\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right) \quad \text{and}  \quad \frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right)$



When $\lambda^{2}|\beta_{j}| \rightarrow 0$ ,  the mode, the mean and the variance of conditional posterior of $\frac{1}{v_{j}}$ and $\frac{1}{\tau_{j}^{2}}$ will tend to infinte. In high dimension and very sparse setting, in very rarely case this will lead to the numerical instable problem. Python will report divide by zero encountered in true divide when evaulating $\sqrt{\frac{1}{4\lambda^{2}|\beta_{j}|}}$ or $\frac{1}{{\lambda}^{2}v_{j}|\beta_{j}|}$. 



We see that 

$\pi(v_{j} \mid \beta_{j},\lambda)\propto \pi(\beta \mid \lambda, v_{j})\pi(v_{j}) \propto \exp\left(-\frac{\lambda^{2}|\beta_{j}|}{v_{j}}-\frac{1}{4}v_{j}\right)v_{j}^{-1/2}$ and $\pi(\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j}) \propto \pi(\beta_{j} \mid \tau_{j}^{2},\lambda)\pi(\tau_{j}^{2}\mid v_{j}) \propto \tau_{j}^{-1}\exp \left(-\frac{\lambda^{4}\beta_{j}^{2}}{\tau_{j}^{2}}-\frac{\tau_{j}^{2}}{2v_{j}^{2}}\right)$



If $\lambda^{2}|\beta_{j}| \rightarrow 0$,  then we have $\pi(v_{j} \mid \beta_{j},\lambda) \rightarrow \mathrm{Gamma}(\frac{1}{2},\frac{1}{4})$ and $\pi(\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j}) \rightarrow \mathrm{Gamma}(\frac{1}{2},\frac{1}{2v_{j}^{2}})$. 

Thus to improve numerical stability,  we define another thresholding parameter $\Delta$.  If $\lambda^{2}|\beta_{j}|<\Delta$,  we will 

Sample  $v_{j} \mid \beta_{j},\lambda \sim \mathrm{Gamma}(\frac{1}{2},\frac{1}{4})$ 

Sample $\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j} \sim \mathrm{Gamma}(\frac{1}{2},\frac{1}{2v_{j}^{2}})$

In practice, we find that, by setting $\Delta \leq 1e^{-5}$,  the resulting approximation error is negligible.



##  PCG sampler for Conjugated $L_\frac{1}{2}$ prior 

Conjugate priors is very popular in Bayesian linear regression. The conjugate prior begins with specifying a prior on $\beta$ that depends on $\sigma$, such that 
$$
\pi\left(\beta \mid \sigma^2\right)=\frac{1}{\sigma^{P}} h(\beta / \sigma)
$$
One of the reason for the popularity of the conjugate prior framework is that it often allows for marginalization over $\beta$ and $\sigma$ , resulting in closed form expressions for Bayes factors and updates of posterior model probabilities. For conjugated $L_{\frac{1}{2}}$ prior, we have
$$
\pi(\beta_{j}\mid \sigma, \lambda) \propto \exp[-\lambda|\beta_{j}/\sigma|^{\frac{1}{2}}]
$$
with the same hyper-prior as before. Then we can construct the PCG sampler:



S1. Sample $\lambda \sim$ Gamma $\left(2P+0.5, \sum_{j=1}^{P} |\sigma\beta_{j}|^{1/2}+1/b \right)$



S2. Sample $b \sim \operatorname{InvGamma~}\left(1,1+\lambda\right)$



S3. Sample $\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{\sigma}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right), \quad j=1, \ldots, p$ 



S4. Sample $\frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{\sigma}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right), \quad j=1, \ldots, p$



S5. Sample $\sigma^{2} \sim \operatorname{InvGamma}\left(\frac{N}{2}, \frac{Y^{T} H^{-1} Y}{2}\right)$



S6. Sample $\beta \sim \mathrm{N}_{P}\left(\left(X^{T} X+ D^{-1}\right)^{-1} X^{T} Y, \sigma^{2}\left(X^{T} X+ D^{-1}\right)^{-1}\right)$



where $H=XDX^{T}+I_{N}$



### Warning 

However, it was argued by this paper https://projecteuclid.org/journals/bayesian-analysis/volume-14/issue-4/Variance-Prior-Forms-for-High-Dimensional-Bayesian-Variable-Selection/10.1214/19-BA1149.full that the use of conjugate shrinkage priors can lead to underestimation of variance in high dimensional linear regression setting.  We also observed this phenomenon in both our Conjugated $L_\frac{1}{2}$ prior and conjugated horseshoe prior. 

In fact, the underestimation of variance also exists for using independent prior for $\beta$, and $\sigma^{2}$ when $N<P$ and $N$ is not large enough. But it will gradually vanish as $N$ and $P$ increase together with some rate.  For the conjugated setting, we never observe the vanish of variance underestimation in high dimesnional setting.

**Although we also provide the code of PCG sampler for conjugated $L_{\frac{1}{2}}$ prior,  we never recommend to use it in high dimensional linear regression problem.** 



## Usage

```
beta_sample,sigma2_sample=Bayesian_L_half_regression(Y,X,M=10000,burn_in=10000) 

beta_mean=np.mean(beta_sample,axis=1)

beta_median=np.median(beta_sample,axis=1)

sigma2_mean=np.mean(sigma2_sample)

sigma2_median=np.median(sigma2_sample)
```



```
beta_sample,sigma2_sample=Conjugated_L_half(Y,X,M=10000,burn_in=10000) 

beta_mean=np.mean(beta_sample,axis=1)

beta_median=np.median(beta_sample,axis=1)

sigma2_mean=np.mean(sigma2_sample)

sigma2_median=np.median(sigma2_sample)
```







$Y$ is the vector of response with length $N$ and $X$ is $N \times P$ covariate matrix. $M$ is the number of the samples from MCMC with default setting 10000. burn_in is the burn in period for MCMC with default setting 10000.  



## Reference

```
@article{johndrow2020scalable,
  title={Scalable approximate MCMC algorithms for the horseshoe prior},
  author={Johndrow, James and Orenstein, Paulo and Bhattacharya, Anirban},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={73},
  year={2020}
}
```

```
@article{bhattacharya2016fast,
  title={Fast sampling with Gaussian scale mixture priors in high-dimensional regression},
  author={Bhattacharya, Anirban and Chakraborty, Antik and Mallick, Bani K},
  journal={Biometrika},
  pages={asw042},
  year={2016},
  publisher={Oxford University Press}
}
```

```
@article{ke2021bayesian,
  title={Bayesian $ L_\frac{1}{2}$ regression},
  author={Ke, Xiongwen and Fan, Yanan},
  journal={arXiv preprint arXiv:2108.03464},
  year={2021}
}
```

```
@article{moran2019variance,
  title={Variance prior forms for high-dimensional Bayesian variable selection},
  author={Moran, Gemma E and Ro{\v{c}}kov{\'a}, Veronika and George, Edward I},
  year={2019}
}
```

