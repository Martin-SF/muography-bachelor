# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import random
import os, sys

from mpl_toolkits.mplot3d.art3d import Line3DCollection

dataset = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

dataset = dataset.reshape(3,3)
dataset
#%%

fig = plt.figure(figsize=(18, 16))
ax = fig.gca(projection='3d')

ax.plot(dataset[:,0], dataset[:,1], dataset[:,2], '.', c='black', markersize=0.1)
#%%

photon = np.logical_or(dataset[:,6] == 0, dataset[:,6] == 3)
electron = dataset[:,6] == 1
positron = dataset[:,6] == 2


ls = dataset[:,0:6].reshape((-1,2,3))

print(ls)

lc_photon = Line3DCollection(ls[photon], linewidths=0.5, colors='b', alpha=0.02)
ax.add_collection(lc_photon)

lc_electron = Line3DCollection(ls[electron], linewidths=1.0, colors='g', alpha=0.95)
ax.add_collection(lc_electron)

lc_positron = Line3DCollection(ls[positron], linewidths=1.0, colors='r', alpha=0.95)
ax.add_collection(lc_positron)


ax.set_xlabel('x [cm]')
ax.set_ylabel('y [cm]')
ax.set_zlabel('z [cm]')