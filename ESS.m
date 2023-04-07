function[Effective_sample_size]=ESS(x)

S=size(x);
variogram=@(t)  sum((x(t+1:end,:)-x(1:(S(1)-t),:)).^2,'all')./(S(2)*(S(1)-t));
post_var = marginal_posterior_var(x);

t = 2;
rho = ones(1,S(1));

negative_autocorr = false;


while (~negative_autocorr) && (t<S(1))

    rho(t)=1- variogram(t) / (2 .* post_var);

    if mod(t,2)
        negative_autocorr = sum(rho(t-1:t)) < 0;
    end

    t=t+1;
   
end

Effective_sample_size=S(1)*S(2)./(1+2*sum(rho(2:t-1)));

end



function[s2]=marginal_posterior_var(x)

S=size(x);

B_over_n = sum((mean(x, 1) - mean(x,'all')).^2,"all") ./ (S(2) - 1);

W = sum((x - repmat(mean(x,1),S(1),1)).^2,'all') ./ (S(2)*(S(1) - 1));

s2 = W .* (S(1) - 1) / S(1) + B_over_n;


end