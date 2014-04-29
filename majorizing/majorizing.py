#!/usr/bin/env python

# Author: Joshua Haas
# Working Under: Gregory Ditzler

# Implementing: Majorizing Interpolation MDS
# Paper: Dimension Reduction and Visualization of Large
#       High-Dimensional Data via Interpolation
# Conference: The ACM International Symposium on High Performance
#       Distributed Computing (HPDC 2010)
# Authors: Seung-Hee Bae, Jon Youl Choi, Judy Qiu, Geoffrey Fox

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

def majorize(old_high,old_low,new_high,k=2,diss=None,epsilon=1e-3,max_iter=25):

    # Calculate pairwise distances if not supplied by the user
    if diss is None:
        old_diss = pairwise_distances(old_high)

    # Find the knn
    knn = NearestNeighbors(n_neighbors=k).fit(old_high)
    (dist, ind) = knn.kneighbors(new_high)

    # Average the knn for each point
    avg = np.array([np.average(old_low[row],0) for row in ind])
    new_low = avg
    
    t = 0
    stress = 0

    # Perform the loop at least once
    # Stop if we reach the STRESS threshold or max_iter
    while (t==0 or (last_stress-stress>epsilon and t<=max_iter)):

        t += 1
        last_stress = stress

        # The following two for loops implement Eq. 17
        # Iterate through every point in the new dataset
        for r in range(0,len(ind)):
            row = avg[r]

            # Iterate through every knn from the old set
            for n in ind[r]:
                
                delta = np.linalg.norm(new_high[r]-old_high[n])
                diff = new_low[r]-old_low[n]
                d = np.linalg.norm(diff)
                row += 1/k*delta/d*diff
                
            new_low[r] = row
        
        union = np.append(old_low,new_low)
        new_diss = pairwise_distances(union)
        stress = np.sum(np.power(new_diss-old_diss,2))
    
    return new_low
