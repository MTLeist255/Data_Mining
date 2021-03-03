# Data Sci Exercise 2:  fitting practice: B: maximum likelihood

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner

def lnlike( theta, x, y, yerr ):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
    return -0.5 * (np.sum((y - model)**2 * inv_sigma2 - np.log( inv_sigma2 )))

def lnprior( theta ):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def lnprob( theta, x, y, yerr ):
    lp = lnprior( theta )
    if not np.isfinite( lp ):
        return -np.inf
    return lp + lnlike( theta, x, y, yerr )

# main code

np.random.seed(123)

# Choose the "true" parameters.

m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.

N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# Plot the data.

plt.ion()
x0 = np.linspace(0, 10, 500)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)

# max likelihood optimization (from 'B')

nll = lambda *args: -lnlike( *args )
result = op.minimize( nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result[ "x" ]
w = m_ml * x0 + b_ml

# Plot the fit: generate additional points as a line.

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, w, "--k", label="LS")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

# Set up for emcee = MCMC

ndim, nwalkers = 3, 100
pos = [ result[ "x" ] + 1e-4 * np.random.randn( ndim ) for i in range( nwalkers )]
sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc( pos, 500)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# add band(s) for MCMC result

xl = np.array([0, 10])
for m, b, lnf in samples[np.random.randint( len(samples), size=100)]:
    plt.plot(xl, m*xl + b, color="k", alpha=0.1)

plt.plot(xl, m_true*xl + b_true, color="r", lw=2, alpha=0.8)
plt.errorbar( x, y, yerr=yerr, fmt=".k")
#plt.savefig( "Fit_MCMC_MaxLike_True.png" )

# look at output of emcee chains

fig = corner.corner( samples, labels = ["$m$", "$b$", "$\ln\,f$"],
                     truths=[m_true, b_true, np.log(f_true)])
#fig.savefig( "Fit_MCMC_triangle.png" )
plt.show()