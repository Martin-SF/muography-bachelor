# %%
# from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
import proposal as pp
from EcoMug.build import EcoMug
import os, sys

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

def calculate_energy(p):
    return np.sqrt(p * p + pp.particle.MuMinusDef().mass**2)

os.chdir(os.path.dirname(sys.argv[0]))
# %%

# 100%|██████████| 5/5 [13:29<00:00, 161.96s/it]
# STATISTICS = int(1e7)
# [1, 10, 50, 100, 1000]

file_name = "EcoMug_muons1.hdf"
energys = {1: 'darkblue', 10: 'indianred', 50: 'c', 100: 'green', 1000: 'm'}
for gen_momentum in tqdm(energys.keys(), disable=False):
    continue
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

    STATISTICS = int(1e6)
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
file_name2 = "EcoMug_muons_1e6.hdf"
file_name2 = "EcoMug_muons2.hdf"
# binsize = 300
binsize = 10

# mu_100gev  = pd.read_hdf(file_name2, key='GeV100')

# linspace = np.linspace(min(data), max(data), binsize)
# linspace = np.linspace(-1, 0, binsize)
# def xy_data(**x, y):
#     return *get_hist_array(np.cos(**x),y)

# linspace = np.linspace(np.pi/2, np.pi, binsize)


def get_hist_array(data, bins):
    # x, y = get_hist_array(data, bins)
    hist = np.histogram(data, bins=bins)
    x = np.delete(hist[1], [(len(hist[1])-1)], None)
    y = hist[0]/hist[0][1]
    return (x, y)

p = 1000
c = 'r'

def change_azimuth_convention(a):
    return -a + np.pi

data = pd.read_hdf(file_name2, key=f'GeV{p}')
theta = np.cos(change_azimuth_convention(data['theta']))
# theta = data['theta']
# hist, bin_edges = np.histogram(theta, bins=binsize)
_ = plt.hist(theta, bins=1000, histtype='step')
# %%
hist
# y, x = np.histogram(data['theta'], bins=linspace)

x = np.delete(bin_edges, [(len(bin_edges)-1)], None)
y = hist/hist[1]
_ = plt.plot(x, y, c, label=f'{p} GeV')



# for p, c in tqdm(energys.items(), disable=False):
    # data = pd.read_hdf(file_name2, key=f'GeV{p}')
    # hist, bin_edges = np.histogram(data['theta'], bins=linspace)
    # # y, x = np.histogram(data['theta'], bins=linspace)
    
    # x = np.delete(bin_edges, [(len(bin_edges)-1)], None)
    # y = hist/hist[1]
    # _ = plt.hist(hist, bins=bin_edges)



    # _ = plt.plot(x, y, c, label=f'{p} GeV')
    # _ = plt.plot(*get_hist_array(
    # np.cos(data['theta']), linspace),
    #  c, label=f'{p} GeV')

# x = linspace  # np.linspace(min(data), max(data), binsize)
# x = np.arccos(linspace)  # np.linspace(min(data), max(data), binsize)

# x = np.linspace(-np.pi, -np.pi/2, binsize)  # np.linspace(min(data), max(data), binsize)
# y = np.cos(x)**2
# _ = plt.plot(np.cos(x), y,'+', label=r'$\mathrm{cos}^2\theta$')

plt.yscale('log')
plt.ylim([1e-2, 1e1])
# plt.xlim([1, 0])
plt.xlabel(r'$\mathrm{cos} \;\theta $')
plt.ylabel(r"$I/I_{\mathrm{vert}}$")
# plt.tight_layout()
plt.legend()
plt.savefig('plot.pdf')


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