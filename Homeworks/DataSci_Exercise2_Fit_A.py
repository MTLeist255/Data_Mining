# Data Sci Exercise 2:  fitting practice

import numpy as np
import matplotlib.pyplot as plt

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

# Set up call to the linear algebra solver.  This works
# because the data are a one-dimensional line.   Call
# solver, then print results.

A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

# Plot the fit: generate additional points as a line.

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
#plt.savefig( "Fit_LeastSqrs.png" )

# generate Pearson correlation coefficient

rho_val = np.corrcoef( x0, y )
print( "rho before fit = ", rho_val )
rho_val = np.corrcoef( x0, w )
print( "rho after fit = ", rho_val )

# generate chi^2 value

chi2 = sum((yresid / yerr)**2)    # sum up
df = len(x) - 2                   # calculate degrees of freedom
print( 'chi2, df = ', chi2, df )
print( ' "p-value": ' )
stats.chi2.cdf( chi2, df )        # generate chi^2 value

plt.show()
