function[beta_sample,sigma2_sample]=L_half(Y,X)

M=10000;
burn_in=10000;
S=size(X);
beta_sample=zeros(S(2),M+burn_in);
sigma2_sample=ones(1,M+burn_in);
tau_sample=ones(S(2),1);
v_sample=ones(S(2),1);
a_sample=1;
lam_sample=1;

w=1;
T1=1e-2;


for i=2:(M+burn_in)

    % Sampling beta
    sigma=sqrt(sigma2_sample(1,i-1));
    D=tau_sample./lam_sample.^2;
    Mask1=D>T1;
    mu=randn([S(2),1]).*D;
    XD=X(:,Mask1).*D(Mask1)';
    omega=XD*XD'./sigma2_sample(1,i-1)+speye(S(1));
    v=omega\(Y./sigma-X*mu./sigma+randn([S(1),1]));
    beta_sample(:,i)=mu;
    beta_sample(Mask1,i)=beta_sample(Mask1,i)+D(Mask1).*XD'*v./sigma;
    

    % Sampling lambda
    lam_sample=gamrnd(2*S(2)+0.5,1./(sum(sqrt(abs(beta_sample(:,i))))+1./a_sample));

    % Sampling a
    a_sample=1./gamrnd(1,1./(1+lam_sample));

    ink=lam_sample.^2.*abs(beta_sample(:,i));

    % Sampling V
    v_sample=2./random('InverseGaussian',1./sqrt(ink),1);

    % Sampling tau
    tau_sample=v_sample./sqrt(random('InverseGaussian',v_sample./ink,1));


    % Sampling sigma2
    err=Y-X*beta_sample(:,i);
    sigma2_sample(i)=1./gamrnd((w+S(1))/2,2./(w+err'*err));

end

beta_sample=beta_sample(:,burn_in+1:end);
sigma2_sample=sigma2_sample(burn_in+1:end);

end

