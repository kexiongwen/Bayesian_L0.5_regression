tic
n=2000;
p=20000;
BetaTrue = zeros(p,1);
BetaTrue(1)=3;
BetaTrue(2)=1.5;
BetaTrue(5)=2;
BetaTrue(10)=1;
BetaTrue(13)=1;
BetaTrue(19)=0.5;
BetaTrue(26)=-0.5;
BetaTrue(31)=2.0;
BetaTrue(46)=-1.2;
BetaTrue(51)=-1;
SigmaTrue=1;
Corr=0.5.^toeplitz((0:p-1));
X=mvnrnd(zeros(1,p),Corr,n);
Y=X*BetaTrue+SigmaTrue.*randn([n 1]);
toc



tic
[beta_sample,sigma2_sample]=L_half_GPU(Y,X);
toc


%tic
%[beta_sample,sigma2_sample]=L_half(Y,X);
%toc


sigma2_mean=mean(sigma2_sample);
beta_mean=mean(beta_sample,2);


[L2,L1,sparsity,Ham,FDR,FNDR,coverage,coverage_nonzero]=metric(beta_sample,sigma2_sample,BetaTrue);

