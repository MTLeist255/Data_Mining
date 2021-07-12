# figure 6.6/6.7)

# FIGURE 6.6) GAUSSIAN MIXTURE MODEL (GMM)

# A two-dimensional mixture of Gaussians for the stellar metallicity data. The left panel shows the number density of
# stars as a function of two measures of their chemical composition: metallicity ([Fe/H]) and alpha-element abundance
# ([alpha/Fe]). The right panel shows the density estimated using mixtures of Gaussians together with the positions
# and covariances (2-sigma levels) of those Gaussians. The center panel compares the information criteria AIC and BIC
# (see Sections 4.3.2 and 5.4.3).

# Author: Jake VanderPlas
# Edited by: Mason Leist
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general

from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture

from astroML.datasets import fetch_sdss_sspp
from astroML.utils.decorators import pickle_results
from astroML.plotting.tools import draw_ellipse
from astroML.datasets import fetch_great_wall

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Get the Segue Stellar Parameters Pipeline data
data = fetch_sdss_sspp(cleaned=True)
X = np.vstack([data['FeH'], data['alphFe']]).T

# truncate dataset for speed
X = X[::5]

#------------------------------------------------------------
# Compute GaussianMixture models & AIC/BIC -> explain GMM
N = np.arange(1, 14)

# eq 6.17
@pickle_results("GMM_metallicity.pkl")
def compute_GaussianMixture(N, covariance_type='full', max_iter=1000):
    models = [None for n in N]
    for i in range(len(N)):
        print(N[i])
        models[i] = GaussianMixture(n_components=N[i], max_iter=max_iter,
                                    covariance_type=covariance_type)
        models[i].fit(X)
    return models

models = compute_GaussianMixture(N)

# middle plot
AIC = [m.aic(X) for m in models]
BIC = [m.bic(X) for m in models]

i_best = np.argmin(BIC)
gmm_best = models[i_best]
print("best fit converged:", gmm_best.converged_)
print("BIC: n_components =  %i" % N[i_best])

#------------------------------------------------------------
# compute 2D density (left-most plot)
FeH_bins = 51
alphFe_bins = 51
H, FeH_bins, alphFe_bins = np.histogram2d(data['FeH'], data['alphFe'],
                                          (FeH_bins, alphFe_bins))

Xgrid = np.array(list(map(np.ravel,
                          np.meshgrid(0.5 * (FeH_bins[:-1]
                                             + FeH_bins[1:]),
                                      0.5 * (alphFe_bins[:-1]
                                             + alphFe_bins[1:]))))).T
log_dens = gmm_best.score_samples(Xgrid).reshape((51, 51))

# #------------------------------------------------------------
# # Plot the results
# fig = plt.figure(figsize=(5, 1.66))
# fig.subplots_adjust(wspace=0.45,
#                     bottom=0.25, top=0.9,
#                     left=0.1, right=0.97)
#
# # plot density
# # a HESS diagram (essentially a 2D histogram) of the [Fe/H] vs [a/Fe] metallicity for a subset of the SEGUE stellar
# # parameters data -> shows 2 distinct clusters in matallicity
#
# ax = fig.add_subplot(131)
# ax.imshow(H.T, origin='lower', interpolation='nearest', aspect='auto',
#           extent=[FeH_bins[0], FeH_bins[-1],
#                   alphFe_bins[0], alphFe_bins[-1]],
#           cmap=plt.cm.binary)
# ax.set_xlabel(r'$\rm [Fe/H]$')
# ax.set_ylabel(r'$\rm [\alpha/Fe]$')
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
# ax.set_xlim(-1.101, 0.101)
# ax.text(0.93, 0.93, "Input",
#         va='top', ha='right', transform=ax.transAxes)
#
# # plot AIC/BIC
# #
# # shows both flavor models w/ 5> components, this is due to the fact that components exist in the background and the background
# # is such that a 2-component model is insufficient to fully describe the data.
# #
# # AIC (Akaike Information Criterion): a simple approach based on an asympototic approximation; the preffered approach
# # to model comparison for the cross-validation, which is based on only the finite data at hand rather than approximations
# # based on infinite data (sec 4.3.2)
# #
# # BIC (Bayesian Information Criterion): attempts to simplify the computation of the odds ratio by making certain assumptions about the likelihood. Easier to
# # # compute and, like the AIC, is based on the max value of the data likelihood (sec 5.4.3)
#
# # When two models are compared, their BIC(or AIC) are compared analogously to the odds ratio, that is, the model with
# # the smaller value wins
# ax = fig.add_subplot(132)
# ax.plot(N, AIC, '-k', label='AIC')
# ax.plot(N, BIC, ':k', label='BIC')
# ax.legend(loc=1)
# ax.set_xlabel('N components')
# plt.setp(ax.get_yticklabels(), fontsize=7)
# plt.title('Figure 6.6) 2D mix of Gaussians for the stellar metallicity data')
#
# # plot best configurations for AIC and BIC
# #
# # Following the BIC, we get N=5. Here is the common confusion w/ GMM: the fact that the info criteria (BIC/AIC) prefer an
# # N-component peak does not necissarily mean that there are N-components. If the clusters in the input data are not near
# # Gaussian, or if there is a strong background, the number of Gaussian components in the mix will not generally correspond
# # to the number of clusters in the data
# ax = fig.add_subplot(133)
# ax.imshow(np.exp(log_dens),
#           origin='lower', interpolation='nearest', aspect='auto',
#           extent=[FeH_bins[0], FeH_bins[-1],
#                   alphFe_bins[0], alphFe_bins[-1]],
#           cmap=plt.cm.binary)
#
# ax.scatter(gmm_best.means_[:, 0], gmm_best.means_[:, 1], c='w')
# for mu, C, w in zip(gmm_best.means_, gmm_best.covariances_, gmm_best.weights_):
#     draw_ellipse(mu, C, scales=[1.5], ax=ax, fc='none', ec='k')
#
# ax.text(0.93, 0.93, "Converged",
#         va='top', ha='right', transform=ax.transAxes)
#
# ax.set_xlim(-1.101, 0.101)
# ax.set_ylim(alphFe_bins[0], alphFe_bins[-1])
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
# ax.set_xlabel(r'$\rm [Fe/H]$')
# ax.set_ylabel(r'$\rm [\alpha/Fe]$')
# plt.show()

# FIGURE 6.7): A two-dimensional mixture of 100 Gaussians (bottom) used to estimate the number density distribution of
# galaxies within the SDSS Great Wall (top). Compare to figures 6.3 and 6.4, where the density for the same
# distribution is computed using both kernel density and nearest-neighbor-based estimates.

# While the underlying density representation is consistent with the distribution of galaxies and the positions of the
# Gaussians themselves correlate with the structure, there is not a one-to-one mapping between the Gaussians and the
# positions of clusters within the data. For these reasons, GMM are often more appropriate when used as a density
# estimator as oppossed to cluster identification.
#------------------------------------------------------------
# load great wall data
X = fetch_great_wall()


#------------------------------------------------------------
# Create a function which will save the results to a pickle file
#  for large number of clusters, computation will take a long time!
@pickle_results('great_wall_GMM.pkl')
def compute_GMM(n_clusters, max_iter=1000, tol=3, covariance_type='full'):
    clf = GaussianMixture(n_clusters, covariance_type=covariance_type,
                          max_iter=max_iter, tol=tol, random_state=0)
    clf.fit(X)
    print("converged:", clf.converged_)
    return clf

#------------------------------------------------------------
# Compute a grid on which to evaluate the result
Nx = 100
Ny = 250
xmin, xmax = (-375, -175)
ymin, ymax = (-300, 200)

Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx),
                                            np.linspace(ymin, ymax, Ny)))).T

#------------------------------------------------------------
# Compute the results
#
# we'll use 100 clusters.  In practice, one should cross-validate
# with AIC and BIC to settle on the correct number of clusters.
#   INCREASE/DECREASE clustering
clf = compute_GMM(n_clusters=100)
log_dens = clf.score_samples(Xgrid).reshape(Ny, Nx)

#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(5, 3.75))
fig.subplots_adjust(hspace=0, left=0.08, right=0.95, bottom=0.13, top=0.9)

ax = fig.add_subplot(211, aspect='equal')
ax.scatter(X[:, 1], X[:, 0], s=1, lw=0, c='k')

ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)

ax.xaxis.set_major_formatter(plt.NullFormatter())
plt.ylabel(r'$x\ {\rm (Mpc)}$')
plt.title('Figure 6.7) SDSS Great Wall Gaussian Mixture')
ax = fig.add_subplot(212, aspect='equal')
ax.imshow(np.exp(log_dens.T), origin='lower', cmap=plt.cm.binary,
          extent=[ymin, ymax, xmin, xmax])
ax.set_xlabel(r'$y\ {\rm (Mpc)}$')
ax.set_ylabel(r'$x\ {\rm (Mpc)}$')
# Un-comment to display plot
plt.show()

