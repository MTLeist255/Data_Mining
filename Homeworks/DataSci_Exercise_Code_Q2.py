# exercise for Data Sci class: Python    Question 2

# The goal for this exercise: a demonstration of several of the concepts that we have been approaching
# in a more theoretical manner (e.g., maximum likelihood estimation). Our overall goal is a comparison of χ2
# minimization (“chi-squared”) versus a Markov Chain Monte Carlo approach. The mathematics between these two
# approaches does not differ in a huge way – in other words, it’s not the chasm between classical and quantum
# physics, for example. But the interpretation of the results does differ. First, let’s implement a simple
# least-squares fitting of a straight line. This will show us the basic fitting approach without creating a
# huge headache (well, that’s the plan!). Throughout this exercise, please keep track of questions you have,
# but divide them into two bins:

# (i) questions about what you are doing or how something works: save for the next class;
# (ii) questions about something that is or appears to be not working:

# 2) Using the Q2 version of the code (Code R Q2.txt, Code Q2.py) repeat the steps of question 1.
# Questions:
# • does you fit still ‘look’ reasonable?
# • why do you think so?
# • what has changed from problem 1? To answer this question, you can compare the two scripts
# (or, on a Linux machine, just ‘diff code for Q1 code for Q2’)
# • what effect(s) did the change have on the slope and intercept?
# • what effect(s) did the change have on the correlation coefficients? the χ2 value?

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats

def lnlike( theta, x, y, yerr ):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0 / ( yerr**2 + model**2 * np.exp(2 * lnf ))
    return -0.5 * (np.sum((y - model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# specify `true' parameters

m_true = -0.9594
b_true = 4.294
f_true = 0.534

# generate synthetic data to fit

N = 50
x = np.sort( 10* np.random.rand(N)) # rand: unif random no. btwn 0 and 1

y = m_true * x + b_true             # generate y data from true values

# generate ultra-small 'errors' for each y value
yerr =  0.001 * np.random.randn(N) # randn: random pull from *normal dist'n*

y = y + yerr   # generate y values with ultra-small scatter

# generate plot

plt.ion()                # turn on interactive plotting
ax = plt.axes()          # builds plot, 0.0 >> 1.0
ax.set_xlim( 0, 10 )     # re-assign x range ...
ax.set_ylim( -8, +8 )    # ... and y range
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.plot( x, y, 'rD', label = 'Synthetic Data' )    # plot data points as red 'Diamond'

# generate least-squares fit to the data 

nll = lambda *args: -lnlike(*args)
result = optimize.minimize( nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result[ "x" ]

print( 'true vs fit values: ' )
print( m_true, b_true, ' -- ', m_ml, b_ml )

# add fitted values and residuals to plot

yfit = m_ml * x + b_ml     # build array of values from fit
ax.plot( x, yfit, 'bo', label = 'Least Squares Fit' )   # plot fitted values as blue 'o'

yresid = y - yfit          # generate residuals from fit
ax.plot( x, yresid, 'g*', label = 'Residuals' ) # plot residuals as green '*'

# generate chi^2 value

chi2 = sum((yresid / yerr)**2)    # sum up
df = len(x) - 2                   # calculate degrees of freedom
print( 'chi2, df = ', chi2, df )
print( ' "p-value": ', stats.chi2.cdf( chi2, df ) ) # generate chi^2 value
plt.legend()
plt.title('Question 2')
#plt.savefig('Question2.png')
plt.show()
