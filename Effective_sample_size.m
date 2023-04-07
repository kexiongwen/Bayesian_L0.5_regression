tic
n=100;
p=1000;
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
Y=X*BetaTrue+SigmaTrue.*normrnd(0,1,[n 1]);
toc


num_chain=10;
S=size(X);
length=10000;

beta_sample=zeros(S(2),length,num_chain);
sigma2_sample=zeros(length,num_chain);

for i=1:num_chain

    [beta_sample(:,:,i),sigma2_sample(:,i)]=L_half_conjugate(Y,X);
    
end

EES_beta_sample=zeros(p,1);

for i=1:p
    EES_beta_sample(i)=ESS(squeeze(beta_sample(i,:,:)));
end
    
     
EES_L_half=mean(EES_beta_sample,'all');
EES_nonzero=mean(EES_beta_sample(BetaTrue~=0),'all');
EES_zero=mean(EES_beta_sample(BetaTrue==0),'all');
  
 