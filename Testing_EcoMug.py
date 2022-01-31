# %%
# from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
# from scipy.stats import norm
import os, sys
import proposal as pp
from EcoMug.build import EcoMug
from numba import jit, njit, prange, vectorize

MU_MINUS_MASS = pp.particle.MuMinusDef().mass

@vectorize(nopython=True)
def change_azimuth_convention(angle_in_rad):
    return -angle_in_rad + np.pi

@vectorize(nopython=True)
def calculate_energy_vectorized_GeV(momentum):
    One_momentum_in_MeV = 1000
    return np.sqrt(momentum * momentum + (MU_MINUS_MASS/One_momentum_in_MeV)**2)

energys = {1: 'darkblue', 10: 'indianred', 50: 'c', 100: 'green', 1000: 'm'}

# os.chdir(os.path.dirname(sys.argv[0]))
# %%
###########  GERERATING MULTIPLE ENERGYS WITH ECOMUG
############################################################
############################################################
############################################################
############################################################

# 100%|██████████| 5/5 [13:29<00:00, 161.96s/it]
# STATISTICS = int(1e7)
# [1, 10, 50, 100, 1000]

file_name = "EcoMug_muons_2-sphere-r-1.hdf"
file_name = "EcoMug_muons_2-sky-width-10.hdf"
for gen_momentum in tqdm(energys.keys(), disable=False):
    # continue
    gen = EcoMug.EcoMug()
    gen.SetUseSky()  # plane surface generation
    gen.SetSkySize((10, 10))  # x and y size of the plane
    # gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center of the plane
    # gen.SetUseHSphere()  # plane Sphere generation
    # gen.SetHSphereRadius(1)
    # gen.SetUseCylinder()
    gen.SetSeed(1909)
    # gen_momentum = 1
    offset = 0
    gen.SetMinimumMomentum(gen_momentum-offset)  # in GeV
    gen.SetMaximumMomentum(gen_momentum-offset)  # in GeV

    STATISTICS = int(1e6)
    muon_pos = [None]*STATISTICS
    muon_p = [float]*STATISTICS
    muon_theta = [float]*STATISTICS
    muon_phi = [float]*STATISTICS
    muon_charge = [float]*STATISTICS

    for event in tqdm(range(STATISTICS), disable=False):
        gen.Generate()
        muon_pos[event]     = gen.GetGenerationPosition()
        muon_p[event]       = gen.GetGenerationMomentum()
        muon_theta[event]   = gen.GetGenerationTheta()
        muon_phi[event]     = gen.GetGenerationPhi()
        muon_charge[event]  = gen.GetCharge()

    muon_e = calculate_energy_vectorized_GeV(muon_p)

    df = pd.DataFrame()
    df['position'] = muon_pos
    df['momentum'] = muon_p
    df['energy'] = muon_e
    df['theta'] = muon_theta
    df['phi'] = muon_phi
    df['charge'] = muon_charge
    df.to_hdf(file_name, key=f'GeV{gen_momentum}')
#%%
%%time
###########  GERERATING WITH ECOMUG
############################################################
############################################################
############################################################
############################################################

gen = EcoMug.EcoMug()
gen.SetUseHSphere()  # plane Sphere generation
gen.SetSeed(1909)

STATISTICS = int(1e6)

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
from concurrent.futures import ThreadPoolExecutor
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
%%time
df['momentum'] = muon_p
df['energy'] = muon_e
df['theta'] = muon_theta
df['phi'] = muon_phi
df['charge'] = muon_charge
# df.to_hdf(file_name, key=f'GeV{gen_momentum}')

# print(muon_e)
# %%
%%time
###########  PLOTTING
############################################################
############################################################
############################################################
############################################################
# file_name2 = file_name
file_name2 = "EcoMug_muons_1e6.hdf"
file_name2 = "EcoMug_muons_2-sky.hdf"
file_name2 = "EcoMug_muons_2-cylinder.hdf"
file_name2 = "EcoMug_muons_2-sphere.hdf"
file_name2 = "EcoMug_muons_2-sphere-r-10000.hdf"
file_name2 = "EcoMug_muons_2-sky-width-10000.hdf"
file_name2 = "EcoMug_muons_2-sky-width-10.hdf"
file_name2 = "EcoMug_muons2.hdf"
file_name2 = "EcoMug_muons_2-sphere-r-1.hdf"
binsize = 100

# mu_100gev  = pd.read_hdf(file_name2, key='GeV100')

# plt.rcParams['figure.figsize'] = (8, 8)
# plt.rcParams['font.size'] = 16
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['axes.labelsize'] = 16

for p, c in tqdm(energys.items(), disable=False):
    data = pd.read_hdf(file_name2, key=f'GeV{p}')
    theta = change_azimuth_convention(np.array(data['theta']))
    cos_theta = np.cos(theta)
    # theta = change_azimuth_convention(np.array(data['theta']))/(np.pi/2)

    hist, bin_edges = np.histogram(cos_theta, bins=binsize)
    bin_edges = np.delete(bin_edges, [(len(bin_edges)-1)], None)
    # hist = hist/max(hist)
    hist = hist/hist[-1]
    bin_length = bin_edges[1] - bin_edges[0]
    _ = plt.plot(bin_edges+bin_length, hist, c, label=f'{p} GeV')
    # _ = plt.hist(theta, bins=binsize, histtype='step')

# x = linspace  # np.linspace(min(data), max(data), binsize)
# x = np.arccos(linspace)  # np.linspace(min(data), max(data), binsize)

x = np.linspace(0, np.pi/2, binsize)
y = np.cos(x)**2
# _ = plt.plot(np.cos(x), y,'--', label=r'$\mathrm{cos}^2\theta$')
# _ = plt.plot(x, y,'--', label=r'$\mathrm{cos}^2\theta$')
y = 1/np.cos(x)
# _ = plt.plot(np.cos(x), y,'--', label=r'$\frac{1}{\mathrm{cos}\theta}$')

plt.yscale('log')
# plt.xlim([np.pi/2, 0])
# plt.ylim(1e-2, 1e1)
plt.axis([1, 0, 1e-2, 1e0])
plt.xlabel(r'$\mathrm{cos} \;\theta $')
plt.ylabel(r"$I/I_{\mathrm{vert}}$")
plt.title(file_name2)
plt.tight_layout()
plt.legend()
# plt.savefig('plot.pdf')

#%%
data_GeV10_theta0 = np.array(pd.read_hdf("EcoMug_muons2.hdf", key=f'GeV10')['theta'])
data_GeV1000_theta = np.array(pd.read_hdf("EcoMug_muons2.hdf", key=f'GeV1000')['theta'])
#%%
%%time
@vectorize(nopython=True)
def ang_trafo(x):
    return change_azimuth_convention(x)
# theta = np.cos(data['theta'])
theta = np.cos(data_GeV10_theta)
data_GeV10_theta = ang_trafo(data_GeV10_theta0)
# theta2 = 
# bins = np.geomspace(min(data)-1, max(data)+1, 10)
# bins = np.cos(np.linspace(min(data_GeV10_theta), max(data_GeV10_theta), 10000))
# bins = np.cos(np.linspace(np.pi/2, np.pi, 10000))
bins = np.linspace(-1, 0, 10000)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel(r'$E \,/\, \mathrm{GeV} $')
plt.xlabel(r'$\;\theta $')
plt.ylabel("Frequency")
# _ = plt.hist(theta, bins=1000, log=True, density=True, stacked = True) 
plt.subplot(121)
# bin2 = np.linspace(min(data_GeV1000_theta), max(data_GeV1000_theta), 10000)
_ = plt.hist(data_GeV10_theta, bins=1000, log=True) 
plt.subplot(122)
_ = plt.hist(np.cos(data_GeV10_theta), bins=1000, log=True, density=True, stacked = True) 
plt.tight_layout()

# plt.axis([1, 0, 1e-2, 1e0])
# _ = 

# %%

# arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# arr = np.array([1,2,3,4])
t = _[1]
t = np.delete(_[1], [9], None)
# _[1] = t
_[0]