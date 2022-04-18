#%%
from multiprocessing import Pool
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
# from scipy.stats import norm
import os, sys
import proposal as pp
from EcoMug.build import EcoMug
from numba import jit, njit, prange, vectorize
from concurrent.futures import ThreadPoolExecutor

MU_MINUS_MASS = pp.particle.MuMinusDef().mass

# @vectorize(nopython=True)
# def change_azimuth_convention(angle_in_rad):
#     return -angle_in_rad + np.pi

# defining dunctions

# @njit(nogil=True)
# def calculate_energy_vectorized_GeV_1(momentum):
#     One_momentum_in_MeV = 1000
#     return np.sqrt(momentum * momentum + (pp.particle.MuMinusDef().mass/One_momentum_in_MeV)**2)
#%%

# @njit(nogil=True)
@vectorize
def calculate_energy_vectorized_GeV(momentum):
    print(1)
    One_momentum_in_MeV = 1000
    return np.sqrt(momentum * momentum + (MU_MINUS_MASS/One_momentum_in_MeV)**2)

#%%
%%time
s = int(1e6)
w = 12*2**13
print(w)
with ThreadPoolExecutor(s/100) as ex:
    ex.map(calculate_energy_vectorized_GeV, np.arange(0, s))

#%%
%%time
s = int(1e6)
muon_p = np.arange(s)
muon_e = calculate_energy_vectorized_GeV(muon_p)

#%%
%%time

def doitagain():
    for i in range(int(1e6)):
        calculate_energy_vectorized_GeV(1)

s = int(1e1)
# muon_p = np.arange(s)
# muon_e = calculate_energy_vectorized_GeV(muon_p)
nm_processes = 2
pool = Pool(processes=nm_processes)
r1 = pool.apply_async(doitagain(), np.arange(s//2))
r2 = pool.apply_async(doitagain(), np.arange(s//2))
pool.close()
pool.join()

#%%
STATISTICS = int(1e6)
gen = EcoMug.EcoMug()
gen.SetUseHSphere()  # plane Sphere generation
gen.SetSeed(1909)


muon_pos = [None]*STATISTICS  # 2,13 - 2,24 s  
# muon_pos = [([float]*3)]*STATISTICS  # same as [None]
# muon_pos = np.empty(STATISTICS*3).reshape(STATISTICS, 3)  dataframe doesnt take array as pos, need to take a len(list)=3 list for storing it as object 
muon_theta = [float]*STATISTICS
muon_phi = [float]*STATISTICS
muon_charge = [float]*STATISTICS
muon_p = np.empty(STATISTICS)
# = np.empty(STATISTICS)  # 5 % slower
# = []  # about as fast as preallocated

# gen.Generate()
# for event in tqdm(range(STATISTICS), disable=True):
#     muon_pos[event]     = gen.GetGenerationPosition()
#     muon_p[event]       = gen.GetGenerationMomentum()
#     muon_theta[event]   = gen.GetGenerationTheta()
#     muon_phi[event]     = gen.GetGenerationPhi()
#     muon_charge[event]  = gen.GetCharge()

# muon_e = calculate_energy_vectorized_GeV(muon_p)
#%%
# @jit(nopython=False)
def generate():
    gen.Generate()
    muon_pos[event]     = gen.GetGenerationPosition()
    muon_p[event]       = gen.GetGenerationMomentum()
    muon_theta[event]   = gen.GetGenerationTheta()
    muon_phi[event]     = gen.GetGenerationPhi()
    muon_charge[event]  = gen.GetCharge()

#%%
%%time
# 12*64*64
with ThreadPoolExecutor(STATISTICS) as ex:
    ex.map(generate(), np.arange(0, STATISTICS))
muon_e = calculate_energy_vectorized_GeV(muon_p)

# df = pd.DataFrame()
# df['position'] = muon_pos
# muon_e = [calculate_energy_vectorized_GeV(p*1e3)/1e3 for p in muon_p]  # 100x slower 

#%%
COUNT = 50000000
def countdown(n):
    while n>0:
        n -= 1

if __name__ == '__main__':
    pool = Pool(processes=2)
    start = time.time()
    r1 = pool.apply_async(countdown, [COUNT//2])
    r2 = pool.apply_async(countdown, [COUNT//2])
    pool.close()
    pool.join()
    end = time.time()
    print('Time taken in seconds -', end - start)
