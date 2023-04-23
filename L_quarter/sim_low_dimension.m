n=1000;
p=70;
SigmaTrue=3.^0.5;
rho=0.5;

[Y,X,BetaTrue]=data_generator2(n,p,SigmaTrue,rho);

tic
[beta_sample,sigma2_sample]=L_quarter_fast(Y,X);
toc

sigma2_mean=mean(sigma2_sample);
beta_mean=mean(beta_sample,2);
[L2,L1,sparsity,Ham,FDR,FNDR,coverage,coverage_nonzero,sigma2]=metric(beta_sample,sigma2_sample,BetaTrue);








