function[beta_sample,sigma2_sample]=L_half_conjugate(Y,X)

M=10000;
burn_in=10000;
S=size(X);
beta_sample=ones(S(2),M+burn_in);
sigma2_sample=ones(1,M+burn_in);
tau_sample=ones(S(2),1);
v_sample=ones(S(2),1);
a_sample=1;
sigma=1;

w=1;
T1=1e-2;
T2=1e-5;
T3=1e-5;

for i=2:(M+burn_in)

    % Sampling lambda
    lam_sample=gamrnd(2*S(2)+0.5,1./(sum(sqrt(abs(beta_sample(:,i-1)/sigma)))+1./a_sample));

    % Sampling a
    a_sample=1./gamrnd(1,1./(1+lam_sample));

    ink=lam_sample.^2.*abs(beta_sample(:,i-1))/sigma;

    % Sampling V
    Mask2=ink<T2;
    v_sample(~Mask2)=2./random('InverseGaussian',1./sqrt(ink(~Mask2)),1);
    v_sample(Mask2)=gamrnd(0.5,4*ones(sum(Mask2),1));

    % Sampling tau
    Mask3=ink<T3;
    tau_sample(~Mask3)=v_sample(~Mask3)./sqrt(random('InverseGaussian',v_sample(~Mask3)./ink(~Mask3),1));
    tau_sample(Mask3)=sqrt(gamrnd(0.5,2*v_sample(Mask3).^2));

    % Sampling sigma2
    D=tau_sample./lam_sample.^2;
    Mask1=D>T1;
    XD=X(:,Mask1).*D(Mask1)';
    omega=XD*XD'+speye(S(1));
    YT_omega_inv_Y=Y'*(omega\Y);
    sigma2_sample(i)=1./gamrnd((w+S(1))/2,2./(w+YT_omega_inv_Y));

    % Sampling beta
    sigma=sqrt(sigma2_sample(1,i));
    mu=randn([S(2),1]).*D;
    v=omega\(Y./sigma-X*mu+randn([S(1),1]));
    beta_sample(:,i)=sigma*mu;
    beta_sample(Mask1,i)=beta_sample(Mask1,i)+sigma.*(D(Mask1).*XD'*v);

end

beta_sample=beta_sample(:,burn_in+1:end);
sigma2_sample=sigma2_sample(burn_in+1:end);

end