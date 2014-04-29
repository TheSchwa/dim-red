#!/usr/bin/env python

import pickle, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("majorizing.pickle","r") as in_file:
    temp = pickle.load(in_file)

old_high = temp["old_data"]
old_map = temp["old_map"]
new_high = temp["new_data"]
new_map = temp["new_map"]
old_low = temp["old_low"]
both_low = temp["both_low"]
new_low = temp["new_low"]
both_low_maj = temp["both_low_maj"]

colors = ["b"]*1000 + ["r"]*967
fig = plt.figure()

ax = fig.add_subplot(221,projection="3d")
ax.scatter(both_low[:,0],both_low[:,1],both_low[:,2],c=colors)

ax = fig.add_subplot(222,projection="3d")
both_low_maj = np.append(old_low,new_low,0)
ax.scatter(both_low_maj[:,0],both_low_maj[:,1],both_low_maj[:,2],c=colors)

ax = fig.add_subplot(223,projection="3d")
ax.scatter(both_low[1000:1967,0],both_low[1000:1967,1],both_low[1000:1967,2],c=["r"]*967)

ax = fig.add_subplot(224,projection="3d")
ax.scatter(new_low[:,0],new_low[:,1],new_low[:,2],c=["r"]*967)

fig.show()

x = input("Enter: ")
