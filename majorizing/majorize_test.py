#!/usr/bin/env python

import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from majorizing import majorize

old_high = np.array([[4,2,6,5,3,6],[5,2,4,7,3,8],[9,4,0,2,2,4],[8,3,9,2,1,6],[4,7,2,5,9,4],[1,0,0,6,9,5]])
old_low = MDS(n_components=3).fit_transform(old_high)

new_high = np.array([[4,7,2,4,6,4],[3,6,9,2,5,6],[1,6,0,4,2,5],[0,8,2,6,8,5],[7,2,2,6,8,3],[1,5,8,3,8,3]])

both_high = np.append(old_high,new_high,0)
both_low = MDS(n_components=3).fit_transform(both_high)

colors = ["b","b","b","b","b","b","r","r","r","r","r","r",]
fig = plt.figure()
ax = fig.add_subplot(121,projection="3d")
ax.scatter(both_low[:,0],both_low[:,1],both_low[:,2],c=colors)

new_low = majorize(old_high,old_low,new_high)

ax = fig.add_subplot(122,projection="3d")
both_low = np.append(old_low,new_low,0)
ax.scatter(both_low[:,0],both_low[:,1],both_low[:,2],c=colors)
fig.show()

x = input("Enter: ")
