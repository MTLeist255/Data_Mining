# FIGURE 3.1)
#
# A representation of the sum of probabilities

# Author: Jake VanderPlas
# Edited by: Mason Leist
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)

# create plot
fig = plt.figure(figsize=(5, 3.75), facecolor='w')
ax = plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
plt.title('Figure 3.1) Sum of Probabilities Representation')

# draw intersecting circles
ax.add_patch(plt.Circle((1.5, 0.2), 3.2, fc='gray', ec='black', alpha=0.5))
ax.add_patch(plt.Circle((-1.5, 0.2), 3.2, fc='gray', ec='black', alpha=0.5))

# add text
text_kwargs = dict(ha='center', va='center', fontsize=12)
ax.text(-2.6, 0.2, "$p(A)$", **text_kwargs)
ax.text(2.6, 0.2, "$p(B)$", **text_kwargs)
ax.text(0.0, 0.2, "$p(A \cap B)$", **text_kwargs)
ax.text(0, -3.3, "$p(A \cup B) = p(A) + p(B) - p(A \cap B)$", **text_kwargs)
ax.text(0, -4.3, "Figure 3.1) Sum of Probabilities", **text_kwargs)
ax.set_xlim(-8, 8)
ax.set_ylim(-6, 6)

plt.show()