import numpy as np


def marginal_posterior_var(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    n_iter,num_chain = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=0) - np.mean(x))**2).sum() / (num_chain - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=0, keepdims=True))**2).sum() / (num_chain*(n_iter - 1))

    # (over) estimate of variance
    s2 = W * (n_iter - 1) / n_iter + B_over_n

    return s2

def ESS(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    n_iter,num_chain= x.shape

    variogram = lambda t: ((x[t:,:] - x[:(n_iter - t),:])**2).sum() / (num_chain * (n_iter - t))

    post_var = marginal_posterior_var(x)

    t = 1
    rho = np.ones(n_iter)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iter):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(num_chain*n_iter/ (1 + 2*rho[1:t].sum()))