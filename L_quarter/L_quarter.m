function[beta_sample,sigma2_sample]=L_quarter(Y,X)


M=10000;
burn_in=10000;
w=1;
S=size(X);
beta_sample=zeros(S(2),M+burn_in);
sigma2_sample=ones(1,M+burn_in);
tau_sample=ones(S(2),1);
v1_sample=ones(S(2),1);
v2_sample=ones(S(2),1);
a_sample=1;
lam_sample=2;
T1=1e-4;
T2=1e-5;



for i=2:(M+burn_in)

    % Sampling beta

    sigma=sqrt(sigma2_sample(1,i-1));
    D=tau_sample./lam_sample.^4;
    Mask1=D>T1;
    mu=normrnd(0,1,[S(2),1]).*D;
    XD=X(:,Mask1).*D(Mask1)';
    s=sum(Mask1);
    disp(s)
    omega=XD*XD'./sigma2_sample(1,i-1)+speye(S(1));
    v=omega\(Y./sigma-X*mu./sigma+normrnd(0,1,[S(1),1]));
    beta_sample(:,i)=mu;
    beta_sample(Mask1,i)=beta_sample(Mask1,i)+D(Mask1).*XD'*v./sigma;

    % Sampling lambda
    lam_sample=gamrnd(4*S(2)+0.5,1./(sum(abs(beta_sample(:,i)).^0.25)+1./a_sample));

    % Sampling a
    a_sample=1./gamrnd(1,1./(1+lam_sample));

    ink1=lam_sample.^4.*abs(beta_sample(:,i));
    ink2=sqrt(ink1)./lam_sample;
    Mask2=ink1<T2;

    % Sampling v2
    v2_sample(~Mask2)=2./random('InverseGaussian',1./ink1(~Mask2).^0.25,1);
    v2_sample(Mask2)=gamrnd(0.5,4*ones(sum(Mask2),1));

    % Sampling v1
    v1_sample(~Mask2)=2*v2_sample(~Mask2).^2./random('InverseGaussian',v2_sample(~Mask2)./ink2(~Mask2),1);
    v1_sample(Mask2)=gamrnd(0.5,4*v2_sample(Mask2).^2);

    % Sampling tau
    tau_sample(~Mask2)=v1_sample(~Mask2)./sqrt(random('InverseGaussian',v1_sample(~Mask2)./ink1(~Mask2),1));
    tau_sample(Mask2)=sqrt(gamrnd(0.5,2*v1_sample(Mask2).^2));

    % Sampling sigma2
    err=Y-X*beta_sample(:,i);
    sigma2_sample(i)=1./gamrnd((w+S(1))/2,2./(w+err'*err));

end

beta_sample=beta_sample(:,burn_in+1:end);
sigma2_sample=sigma2_sample(burn_in+1:end);

end