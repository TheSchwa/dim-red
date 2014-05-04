#!/usr/bin/env python

import pickle, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EmpTools import EmpData

def main():
    with open("majorizing.pickle","r") as in_file:
        temp = pickle.load(in_file)

    old_map = temp["old_map"]
    old_ids = temp["old_ids"]
    new_map = temp["new_map"]
    new_ids = temp["new_ids"]
    old_low = temp["old_low"]
    both_low = temp["both_low"]
    new_low = temp["new_low"]
    both_low_maj = temp["both_low_maj"]

    both_ids = old_ids + new_ids

    write_map(old_map+new_map[1:len(new_map)],"both.map")

    both_low /= abs(both_low).max()
    both_low_maj /= abs(both_low_maj).max()

    write_pca(both_low,both_ids,"both.mds")
    write_pca(both_low_maj,both_ids,"maj.mds")

def write_pca(data,ids,name):
    lines = [(ids[r]+"\t"+"\t".join(map(str,data[r]))+"\n") for r in range(0,len(data))]
    lines.insert(0,"pc vector number\t1\t2\t3\n")
    lines.extend(["\n","\n"])
    lines.append("eigvals\t1.0\t2.0\t3.0\n")
    lines.append("% variation explained\t30.0\t20.0\t10.0")
    with open(name,"w") as out_file:
        out_file.writelines(lines)

def write_map(lines,name):
    lines = [l+"\n" for l in lines]
    lines[0] = "#" + lines[0]
    with open(name,"w") as out_file:
        out_file.writelines(lines)
    
if __name__ == "__main__":
    main()
