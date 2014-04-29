#!/usr/bin/env python

import pickle, time
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from majorizing import majorize

print("Unpickling...")
with open("majorizing.pickle","r") as in_file:
    temp = pickle.load(in_file)

print("Processing old...")
start = time.time()
old_high = temp["old_data"]
old_map = temp["old_map"]
old_low = MDS(n_components=3).fit_transform(old_high)
print("Old took "+str(time.time()-start)+" seconds.")

print("Processing new...")
new_high = temp["new_data"]
new_map = temp["new_map"]

print("Processing both...")
start = time.time()
both_high = np.append(old_high,new_high,0)
both_low = MDS(n_components=3).fit_transform(both_high)
print("Both took "+str(time.time()-start)+" seconds.")

print("Plotting both...")
colors = ["b"]*1000 + ["r"]*967
fig = plt.figure()
ax = fig.add_subplot(221,projection="3d")
ax.scatter(both_low[:,0],both_low[:,1],both_low[:,2],c=colors)

print("Majorizing...")
start = time.time()
new_low = majorize(old_high,old_low,new_high)
print("Majorizing took "+str(time.time()-start)+" seconds.")

print("Plotting Interpolation...")
ax = fig.add_subplot(222,projection="3d")
both_low_maj = np.append(old_low,new_low,0)
ax.scatter(both_low_maj[:,0],both_low_maj[:,1],both_low_maj[:,2],c=colors)

fig.show()

temp["old_low"] = old_low
temp["both_low"] = both_low
temp["new_low"] = new_low
temp["both_low_maj"] = both_low_maj

with open("majorizing.pickle","w") as out_file:
    pickle.dump(temp,out_file,-1)

x = input("Enter: ")
