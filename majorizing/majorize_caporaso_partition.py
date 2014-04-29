#!/usr/bin/env python

import random, pickle
from EmpTools import EmpData
from sklearn.manifold import MDS

ed = EmpData()
ed.load_biom("caporaso.biom")
ed.load_map("caporaso.map")

old_i = sorted(random.sample(range(0,1967),1000))
new_i = [i for i in range(0,1967) if i not in old_i]

old_data = ed.data[old_i]
old_map = []
for row in old_i:
    sample = ed.biom["columns"][row]["id"]
    map_row = [r for (r,meta) in enumerate(ed.mapping) if sample in meta][0]
    old_map.append(ed.mapping[map_row])

new_data = ed.data[new_i]
new_map = []
for row in new_i:
    sample = ed.biom["columns"][row]["id"]
    map_row = [r for (r,meta) in enumerate(ed.mapping) if sample in meta][0]
    new_map.append(ed.mapping[map_row])

temp = {"old_data":old_data,"old_map":old_map,"new_data":new_data,"new_map":new_map}
with open("majorizing.pickle", "w") as out_file:
    pickle.dump(temp,out_file,-1)
