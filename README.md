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

We develop the PCG sampler that targets the exact horseshoe posterior with the following update rule:



S1. Sample $\beta \sim \mathrm{N}_{P}\left(\left(X^{T} X+\sigma^2 \lambda^{4} D_{\tau^2}^{-1}\right)^{-1} X^{T} Y, \sigma^2\left(X^T X+\sigma^2 \lambda^{4} D_{\tau^2}^{-1}\right)^{-1}\right)$



S2. Sample $\lambda \sim$ Gamma $\left(2 p+0.5, \sum_{j=1}^p\left|\beta_{j}\right|^{\frac{1}{2}}+\frac{1}{b}\right)$



S3. Sample $\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right), \quad j=1, \ldots, p$ 



S4. Sample $\frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right), \quad j=1, \ldots, p$



S5. Sample $\sigma^{2} \sim \operatorname{InvGamma}\left(\frac{n}{2}, \frac{1}{2}\left(Y-X \beta\right)^T\left(Y-X \beta\right)\right)$



S6. Sample $b \sim \operatorname{InvGamma~}\left(1,1+\lambda\right)$



where $D_{\tau^{2}}=\mathrm{Diag}(\tau^{2})$. **Note that unlike the traditional Gibbs sampler, some of the order for the update does matter in PCG sampler!!!**


The efficient method from https://arxiv.org/pdf/1506.04778.pdf has been used to sample from the conditional posterior $\pi(\beta \mid \sigma^{2}, \lambda, \tau^2)$




1. Sample $u  \sim \mathrm{N}\left(0, \lambda^{-4} D_{\tau^{2}}\right)$ and $f \sim \mathrm{N}\left(0, I_{N}\right)$

2. $M=\frac{XD_{\tau^{2}}X^{T}}{\sigma\lambda^{4}}+I_{N}$

3. Set   $v =\frac{Xu}{\sigma}+f$ and $v^{*}=M^{-1}(Y / \sigma-v)$

4. Set  $\beta=u+\frac{1}{\sigma\lambda^{4}}D_{\tau^{2}}X^{T}v^{*}$

   

## JOB-approximation

To further  the reduce computational cost per step, we employ the strategy from  https://jmlr.org/papers/v21/19-536.html  to approximate the matrix product $\frac{XD_{\tau^{2}}X^{T}}{\lambda^{4}}$ by hard-thresholding , resulting in


$$
M \approx I_{N}+\frac{1}{\sigma}XD_{\delta}X, \quad D_{\delta}=\frac{1}{\lambda^{4}}\mathrm{Diag}(\tau_{j}^{2}1({\tau_{j}^{2}/\lambda^{4}>\delta}))
$$


for “small” $\delta$. Using this strategy,the approximate algorithm uses the same update rule as in the MCMC scheme above, with only two changes:



1. $M$ is replaced by $M_{\delta}$ everywhere that it appears in the PCG sampler; and

2. In the final step of sampling $\pi(\beta \mid \sigma^{2}, \lambda, \tau^2)$, the quantity $\frac{1}{\lambda^{4}}D_{\tau^{2}} X^{T}$ is replaced by $D_\delta X^{T}$



Finally, we set the hard-threshold  $\delta=1e-4$ as suggested by their paper.



## Extra approximation to improve the numerical stability 

In PCG sampler scheme for $L_{1/2}$ prior, we need to sample 



$\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right) \quad \text{and}  \quad \frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right)$



When $\lambda^{2}|\beta_{j}| \rightarrow 0$ ,  the mode, the mean and the variance of conditional posterior of $\frac{1}{v_{j}}$ and $\frac{1}{\tau_{j}^{2}}$ will tend to infinte. In high dimension and very sparse setting, in very rarely case this will lead to the numerical instable problem. Python will report divide by zero encountered in true divide when evaulating $\sqrt{\frac{1}{4\lambda^{2}|\beta_{j}|}}$ or $\frac{1}{{\lambda}^{2}{v}_{j}|\beta_{j}|}$. 



We see that 

$\pi(v_{j} \mid \beta_{j},\lambda)\propto \pi(\beta \mid \lambda, v_{j})\pi(v_{j}) \propto \exp\left(-\frac{\lambda^{2}|\beta_{j}|}{v_{j}}-\frac{1}{4}v_{j}\right)v_{j}^{-1/2}$ and $\pi(\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j}) \propto \pi(\beta_{j} \mid \tau_{j}^{2},\lambda)\pi(\tau_{j}^{2}\mid v_{j}) \propto \tau_{j}^{-1}\exp \left(-\frac{\lambda^{4}\beta_{j}^{2}}{\tau_{j}^{2}}-\frac{\tau_{j}^{2}}{2v_{j}^{2}}\right)$



If $\lambda^{2}|\beta_{j}| \rightarrow 0$,  then we have $\pi(v_{j} \mid \beta_{j},\lambda) \rightarrow \mathrm{Gamma}(\frac{1}{2},\frac{1}{4})$ and $\pi(\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j}) \rightarrow \mathrm{Gamma}(\frac{1}{2},\frac{1}{2v_{j}^{2}})$. 

Thus to improve numerical stability,  we define another thresholding parameter $\Delta$.  If $\lambda^{2}|\beta_{j}|<\Delta$,  we will 

Sample  $v_{j} \mid \beta_{j},\lambda \sim \mathrm{Gamma}(\frac{1}{2},\frac{1}{4})$ 

Sample $\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j} \sim \mathrm{Gamma}(\frac{1}{2},\frac{1}{2v_{j}^{2}})$

In practice, we find setting $\Delta \leq 1e^{-5}$  the resulting approximation error is negligible.



## Usage

```
beta_sample,sigma2_sample=Bayesian_L0.5_regression(Y,X,M=10000,burn_in=10000) 

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

