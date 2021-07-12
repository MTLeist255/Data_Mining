# Here is a practice set of data to demonstrate the Gonzalez ALgorithm in 6.4.3). Once the base code is running,
# we'll adapt this code to run with the SDSS data from Figure 6.6) and 6.13). Code referrenced from
# https://www.kaggle.com/barelydedicated/gonzalez-algorithm?scriptVersionId=2656948

from __future__ import print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.spatial import distance
import seaborn as sns
import matplotlib as mpl
import math
import os
from astroML.datasets import fetch_sdss_sspp

np.random.seed(0)
data = fetch_sdss_sspp(cleaned=True)

# cut out some additional strange outliers-> DEMO CHANGES
#data = data[~((data['alphFe'] > 0.4) & (data['FeH'] > -0.3))]

X = np.double(np.vstack([data['FeH'], data['alphFe']]).T)

# Gonzalez algorithm math ->
def gonzalez(data, cluster_num, technique = 'max'):
    clusters = []
    clusters.append(data[0]) # let us assign the first cluster point to be first point of the data
    while len(clusters) != cluster_num:
        if technique == 'max':
            clusters.append(max_dist(data, clusters))
        if technique == 'norm':
            clusters.append(norm_dist(data, clusters))
        # we add the furthest point from ALL current clusters
    return (clusters)

# COMMENT THIS -> Explain coding in order (max-dist undefined up until here)
# cluster_points = gonzalez(X, 3)
# A = np.array(cluster_points)
# print('test', A)

def max_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for cluster_id, cluster in enumerate(clusters):
        for point_id, point in enumerate(data):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster (obselete)
            if not math.isinf(distances[point_id]):
                # if a point is not obselete, then we add the distance to its specific bin
                distances[point_id] = distances[point_id] + distance.euclidean(point,cluster)
                # return the point which is furthest away from all the other clusters
    return data[np.argmax(distances)]

# cluster point works here
cluster_points = gonzalez(X, 3)
A = np.array(cluster_points)
print('test', A)

cluster_points = gonzalez(X, 3)
print('Cluster Centeroids (N = 3):', cluster_points)

cluster_distance = np.full(len(X), np.inf)
for point_idx, point in enumerate(X):
    for cluster_idx, cluster_point in enumerate(cluster_points):
        if cluster_distance[point_idx] == math.inf:
            cluster_distance[point_idx] = distance.euclidean(point,cluster_point)
            continue
        if distance.euclidean(point,cluster_point) < cluster_distance[point_idx]:
            cluster_distance[point_idx] = distance.euclidean(point,cluster_point)
print('3-center cost:', np.max(cluster_distance))

cost = math.sqrt(np.sum(cluster_distance**2, axis=0) /len(X))
print('3-means cost:', cost)

plt.scatter(data['FeH'],data['alphFe'])
for index, point in enumerate(cluster_points):
    if index != 0: # these points are generated
        plt.scatter(point[0],point[1], marker='*', c='red', s=50)
    if index == 0: # this is our ifrst point, which was picked staticly
        plt.scatter(point[0],point[1], marker='*', c='orange', s=50)

cluster_points = gonzalez(X, 4)
print('Cluster Centeroids (N = 4):', cluster_points)

# plt.scatter(data['FeH'],data['alphFe'])
# for index, point in enumerate(cluster_points):
#     if index is not 0: # these points are generated
#         plt.scatter(point[0],point[1], marker='*', c='red', s=50)
#     if index is 0: # this is our ifrst point, which was picked staticly
#         plt.scatter(point[0],point[1], marker='*', c='orange', s=50)

def norm_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for point_id, point in enumerate(data):
        for cluster_id, cluster in enumerate(clusters):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster (obselete)
            if not math.isinf(distances[point_id]):
                # if a point is not obselete, then we add the distance to its specific bin
                distances[point_id] = distances[point_id] + math.pow(distance.euclidean(point,cluster),2)
                # return the point which is furthest away from all the other clusters
    for distance_id, current_distance in enumerate(distances):
        if not math.isinf(current_distance):
            distances[distance_id] = math.sqrt(current_distance/len(data))
    return data[np.argmax(distances)]
#
# cluster_points = gonzalez(X, 3,'norm')
# print('Cluster Centeroids (N = 3):', cluster_points)
#
# # plt.scatter(data['FeH'],data['alphFe'])
# # for index, point in enumerate(cluster_points):
# #     if index is not 0: # these points are generated
# #         plt.scatter(point[0],point[1], marker='*', c='red', s=50)
# #     if index is 0: # this is our ifrst point, which was picked staticly
# #         plt.scatter(point[0],point[1], marker='*', c='orange', s=50)
#
# # Visualize the results
# H, FeH_bins, alphFe_bins = np.histogram2d(data['FeH'], data['alphFe'], 50)
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot()
#
# # plot density
# ax = plt.axes()
# ax.imshow(H.T, origin='lower', interpolation='nearest', aspect='auto',
#           extent=[FeH_bins[0], FeH_bins[-1],
#                   alphFe_bins[0], alphFe_bins[-1]],
#           cmap=plt.cm.binary)
#
# # plot cluster centers
# cluster_points = gonzalez(X, 4)
# #cluster_centers = scaler.inverse_transform(clf.cluster_centers_)
# ax.scatter(cluster_points[:][0], cluster_points[:][1],
#            s=40, c='w', edgecolors='k')
#
# # plot cluster boundaries
# FeH_centers = 0.5 * (FeH_bins[1:] + FeH_bins[:-1])
# alphFe_centers = 0.5 * (alphFe_bins[1:] + alphFe_bins[:-1])
#
# Xgrid = np.meshgrid(FeH_centers, alphFe_centers)
# Xgrid = np.array(Xgrid).reshape((2, 50 * 50)).T
#
# #H = clf.predict(scaler.transform(Xgrid)).reshape((50, 50))
#
# for i in range(cluster_points):
#     Hcp = H.copy()
#     flag = (Hcp == i)
#     Hcp[flag] = 1
#     Hcp[~flag] = 0
#
#     ax.contour(FeH_centers, alphFe_centers, Hcp, [-0.5, 0.5],
#                linewidths=1, colors='k')
#
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
# ax.set_xlim(-1.101, 0.101)
# ax.set_ylim(alphFe_bins[0], alphFe_bins[-1])
#
# ax.set_xlabel(r'$\rm [Fe/H]$')
# ax.set_ylabel(r'$\rm [\alpha/Fe]$')
# plt.title('Figure 6.13) K-means analysis of the stellar\n metallicity data from Fig 6.6)')
# plt.show()