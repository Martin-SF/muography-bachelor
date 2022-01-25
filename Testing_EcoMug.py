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



def calculate_energy(p):
    return np.sqrt(p * p + pp.particle.MuMinusDef().mass**2)

def change_azimuth_convention(a):
    return -a + np.pi

energys = {1: 'darkblue', 10: 'indianred', 50: 'c', 100: 'green', 1000: 'm'}

os.chdir(os.path.dirname(sys.argv[0]))

# %%
############################################################
############################################################
###########  GERERATING MULTIPLE ENERGYS WITH ECOMUG
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
#%%
############################################################
############################################################
###########  GERERATING WITH ECOMUG
############################################################
############################################################

%%time

# continue
# file_name = "EcoMug_muons_all_1e7.hdf"
gen = EcoMug.EcoMug()
gen.SetUseHSphere()  # plane Sphere generation
gen.SetSeed(1909)

STATISTICS = int(1e7)
# muon_position = []
# muon_p = []
# muon_theta = []
# muon_phi = []
# muon_charge = []
# muon_position = [([float]*3)]*STATISTICS
# muon_p = [float]*STATISTICS
# muon_theta = [float]*STATISTICS
# muon_phi = [float]*STATISTICS
# muon_charge = [float]*STATISTICS
muon_p = np.empty(STATISTICS)
muon_theta = np.empty(STATISTICS)
muon_phi = np.empty(STATISTICS)
muon_charge = np.empty(STATISTICS)
muon_e = np.empty(STATISTICS)

gen.Generate()
for event in tqdm(range(STATISTICS), disable=False):
    # muon_position.append(gen.GetGenerationPosition())
    # muon_p.append(gen.GetGenerationMomentum())
    # muon_theta.append(gen.GetGenerationTheta())
    # muon_phi.append(gen.GetGenerationPhi())
    # muon_charge.append(gen.GetCharge())

    # muon_position[event]= gen.GetGenerationPosition()
    muon_p[event]       = gen.GetGenerationMomentum()
    # muon_e[event]       = calculate_energy(muon_p[event]*1e3)/1e3
    muon_theta[event]   = gen.GetGenerationTheta()
    muon_phi[event]     = gen.GetGenerationPhi()
    muon_charge[event]  = gen.GetCharge()


muon_e = [calculate_energy(p*1e3)/1e3 for p in muon_p]
#%%
df = pd.DataFrame()
df['position'] = muon_position
df['momentum'] = muon_p
df['energy'] = muon_e
df['theta'] = muon_theta
df['phi'] = muon_phi
df['charge'] = muon_charge
# df.to_hdf(file_name, key=f'GeV{gen_momentum}')

# print(muon_e)
# %%
############################################################
############################################################
###########  PLOTTING
############################################################
############################################################
# file_name2 = file_name
file_name2 = "EcoMug_muons2.hdf"
file_name2 = "EcoMug_muons_1e6.hdf"
file_name2 = "EcoMug_muons_2-sky.hdf"
file_name2 = "EcoMug_muons_2-cylinder.hdf"
file_name2 = "EcoMug_muons_2-sphere.hdf"
file_name2 = "EcoMug_muons_2-sphere-r-10000.hdf"
file_name2 = "EcoMug_muons_2-sphere-r-1.hdf"
file_name2 = "EcoMug_muons_2-sky-width-10000.hdf"
file_name2 = "EcoMug_muons_2-sky-width-10.hdf"
binsize = 100

# mu_100gev  = pd.read_hdf(file_name2, key='GeV100')

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 16

for p, c in tqdm(energys.items(), disable=False):
    data = pd.read_hdf(file_name2, key=f'GeV{p}')
    theta = np.cos(change_azimuth_convention(data['theta']))
    # theta = data['theta']
    hist, bin_edges = np.histogram(theta, bins=binsize)
    bin_edges = np.delete(bin_edges, [(len(bin_edges)-1)], None)
    hist = hist/hist[-1]
    _ = plt.plot(bin_edges, hist, c, label=f'{p} GeV')
    # _ = plt.hist(theta, bins=binsize, histtype='step')

# x = linspace  # np.linspace(min(data), max(data), binsize)
# x = np.arccos(linspace)  # np.linspace(min(data), max(data), binsize)

x = np.linspace(0, np.pi/2, binsize)
y = np.cos(x)**2
_ = plt.plot(np.cos(x), y,'--', label=r'$\mathrm{cos}^2\theta$')
y = 1/np.cos(x)
_ = plt.plot(np.cos(x), y,'--', label=r'$\frac{1}{\mathrm{cos}\theta}$')

plt.yscale('log')
plt.xlim([1, 0])
plt.ylim([1e-2, 1e1])
plt.xlabel(r'$\mathrm{cos} \;\theta $')
plt.ylabel(r"$I/I_{\mathrm{vert}}$")
plt.title(file_name2)
# plt.tight_layout()
plt.legend()
# plt.savefig('plot.pdf')


# %%
data = pd.read_hdf(file_name2, key=f'GeV{p}')['energy']
# bins = np.geomspace(min(data)-1, max(data)+1, 10)
plt.xscale('log')
plt.xlabel(r'$E \,/\, \mathrm{GeV} $')
plt.ylabel("Frequency")
_ = plt.hist(data, bins=10, log=True) 
# _ = 

# %%

# arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# arr = np.array([1,2,3,4])
t = _[1]
t = np.delete(_[1], [9], None)
# _[1] = t
_[0]