# FIGURE 1.5/1.6):
#
# FIGURE 1.5): from Stripe 82-> SDSS Segue Stellar Parameter Pipeline Data: The surface
# gravity vs. effective temperature plot for the first 10,000 entries from the catalog of
# stars with SDSS spectra. The rich substructure reflects both stellar physics and the
# SDSS selection criteria for spectroscopic follow-up. The plume of points centered on
# Teff ~ 5300 K and log g ~ 3 is dominated by red giant stars, and the locus of points
# with Teff < 6500 K and log g > 4.5 is dominated by main sequence stars. Stars to the
# left from the main sequence locus are dominated by the so-called blue horizontal
# branch stars. The axes are plotted backward for ease of comparison with the
# classical Hertzsprung-Russell diagram: the luminosity of a star approximately
# increases upward in this diagram.

# Author: Jake VanderPlas
# Edited by: Mason Leist
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_sspp
from astroML.datasets import fetch_sdss_S82standards

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Fetch the data
data = fetch_sdss_sspp()

# select the first 10000 points
data = data[:10000]

# do some reasonable magnitude cuts
rpsf = data['rpsf']
data = data[(rpsf > 15) & (rpsf < 19)]

# get the desired data
logg = data['logg']
Teff = data['Teff']

#------------------------------------------------------------
# Plot the data
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(Teff, logg, marker='.', markersize=2, linestyle='none', color='black')

ax.set_xlim(8000, 4500)
ax.set_ylim(5.1, 1)

ax.set_xlabel(r'$\mathrm{T_{eff}\ (K)}$')
ax.set_ylabel(r'$\mathrm{log_{10}[g / (cm/s^2)]}$')
plt.title('Figure 1.5) SDSS Segue Stellar Parameter Pipeline Data')

# FIGURE 1.6): from Stripe 82-> SDSS Stripe 82 Standard Stars: The
# g-r vs. r-i color-color diagram for the first 10,000 entries in the
# Stripe 82 Standard Star Catalog. The region with the highest point
# density is dominated by main sequence stars. The thin extension
# toward the lower-left corner is dominated by the so-called blue
# horizontal branch stars and white dwarf stars.

# Author: Jake VanderPlas
# Edited by: Mason Leist

# Fetch the stripe 82 data
data = fetch_sdss_S82standards()

# select the first 10000 points
data = data[:10000]

# select the mean magnitudes for g, r, i
g = data['mmu_g']
r = data['mmu_r']
i = data['mmu_i']

#------------------------------------------------------------
# Plot the g-r vs r-i colors
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(g - r, r - i, marker='.', markersize=2,
        color='black', linestyle='none')

ax.set_xlim(-0.6, 2.0)
ax.set_ylim(-0.6, 2.5)

ax.set_xlabel(r'${\rm g - r}$')
ax.set_ylabel(r'${\rm r - i}$')
plt.title('Figure 1.6) SDSS Stripe 82 Standard Stars')
plt.show()