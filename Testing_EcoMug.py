# %%
# from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import random
import pandas as pd
import proposal as pp
from EcoMug.build import EcoMug

def get_hist_array(data, bins):
    # ax = plt.subplot()
    # hist_output = ax.hist(data, bins=bins)
    # hist_output = ax.hist(data, bins=bins)
    # ax.close()
    hist_output = np.histogram(data, bins=bins)
    return ((np.delete(hist_output[1], [(len(hist_output[1])-1)], None)), 
            (hist_output[0]/hist_output[0][0]))


plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

def calculate_energy(p):
    return np.sqrt(p * p + pp.particle.MuMinusDef().mass**2)

# %%

# 100%|██████████| 5/5 [13:29<00:00, 161.96s/it]
# STATISTICS = int(1e7)
# [1, 10, 50, 100, 1000]

file_name = "EcoMug_muons3.hdf"
for gen_momentum in tqdm([1, 10, 50, 100, 1000], disable=False):
    gen = EcoMug.EcoMug()
    # gen.SetUseSky()  # plane surface generation
    gen.SetUseHSphere()  # plane Sphere generation
    # gen.SetSkySize((10, 10))  # x and y size of the plane
    # gen.SetSkyCenterPosition((0, 0, 20))  # (x,y,z) position of the center of the plane
    gen.SetSeed(1909)
    # gen_momentum = 1
    offset = 0
    gen.SetMinimumMomentum(gen_momentum-offset)  # in GeV
    gen.SetMaximumMomentum(gen_momentum-offset)  # in GeV

    STATISTICS = int(1e7)
    muon_position = []
    muon_p = []
    muon_theta = []
    muon_phi = []
    muon_charge = []

    for event in tqdm(range(STATISTICS), disable=False):
        gen.Generate()

        muon_position.append(gen.GetGenerationPosition())
        muon_p.append(gen.GetGenerationMomentum())
        muon_theta.append(gen.GetGenerationTheta())
        muon_phi.append(gen.GetGenerationPhi())
        muon_charge.append(gen.GetCharge())

    muon_e = [calculate_energy(p*1e3)/1e3 for p in muon_p]

    df = pd.DataFrame()
    df['position'] = muon_position
    df['momentum'] = muon_p
    df['energy'] = muon_e
    df['theta'] = muon_theta
    df['phi'] = muon_phi
    df['charge'] = muon_charge
    df.to_hdf(file_name, key=f'GeV{gen_momentum}')
# print(muon_e)
# %%
# file_name2 = file_name
file_name2 = "EcoMug_muons2.hdf"
mu_1gev    = pd.read_hdf(file_name2, key='GeV1')
mu_10gev   = pd.read_hdf(file_name2, key='GeV10')
mu_50gev   = pd.read_hdf(file_name2, key='GeV50')
mu_100gev  = pd.read_hdf(file_name2, key='GeV100')
mu_1000gev = pd.read_hdf(file_name2, key='GeV1000')

# data = np.cos(mu_1gev['theta']+np.pi)
data = np.cos(mu_1gev['theta'])
# bins = np.linspace(min(data), max(data), 100)
# x, y = get_hist_array(data, bins)
binsize = 100
    # plt.plot(*get_hist_array(
    #     np.cos(mu_1gev['theta']+np.pi), np.linspace(min(data), max(data), binsize)),
    #      'r', label='1GeV')
    # plt.plot(*get_hist_array(
    #     np.cos(mu_10gev['theta']+np.pi), np.linspace(min(data), max(data), binsize)),
    #      'b', label='10GeV')
    # plt.plot(*get_hist_array(
    #     np.cos(mu_100gev['theta']+np.pi), np.linspace(min(data), max(data), binsize)),
    #      'g', label='100GeV')
    # plt.plot(*get_hist_array(
    #     np.cos(mu_1000gev['theta']+np.pi), np.linspace(min(data), max(data), binsize)),
    #      'm', label='1000GeV')

_ = plt.plot(*get_hist_array(
    np.cos(mu_1gev['theta']), np.linspace(min(data), max(data), binsize)),
     'r', label='1GeV')
_ = plt.plot(*get_hist_array(
    np.cos(mu_10gev['theta']), np.linspace(min(data), max(data), binsize)),
     'b', label='10GeV')
_ = plt.plot(*get_hist_array(
np.cos(mu_50gev['theta']), np.linspace(min(data), max(data), binsize)),
'c', label='50GeV')
_ = plt.plot(*get_hist_array(
    np.cos(mu_100gev['theta']), np.linspace(min(data), max(data), binsize)),
     'g', label='100GeV')
_ = plt.plot(*get_hist_array(
    np.cos(mu_1000gev['theta']), np.linspace(min(data), max(data), binsize)),
     'm', label='1000GeV')


x = np.linspace(min(data), max(data), binsize)
y = np.cos(x+1)**2
_ = plt.plot(x, y,',', label=r'$\mathrm{cos}^2\theta$')

plt.yscale('log')
plt.ylim([1e-2, 1.1])
# plt.xlim([1, 0])
plt.xlabel(r'$\mathrm{cos} \theta $')
plt.ylabel(r"$I/I_{\mathrm{vert}}$")
plt.legend()



# %%
data = df['energy']
bins = np.geomspace(min(data)-1, max(data)+1, 10)
plt.xscale('log')
plt.xlabel(r'$E \,/\, \mathrm{GeV} $')
plt.ylabel("Frequency")
_ = plt.hist(data, bins=bins, log=True) 
# _ = 

# %%

# arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# arr = np.array([1,2,3,4])
t = _[1]
t = np.delete(_[1], [9], None)
# _[1] = t
_[0]