# %%
# from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
# from scipy.stats import norm
# import proposal as pp
# from EcoMug_pybind11.build import EcoMug
import EcoMug
# from numba import vectorize
# from numba import jit, njit, prange
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib

slib.change_zenith_convention(0)
slib.calculate_energy_vectorized_GeV(0)

# %
# GERERATING full spectra muons ECOMUG
############################################################
############################################################
############################################################
############################################################
t = stopwatch.stopwatch(title='generating ecomug muons', selfexecutiontime_micros=0, time_unit='ms')
t.task('generating ecomug object')
gen = EcoMug.EcoMug()
# gen.SetUseHSphere()  # plane Sphere generation
gen.SetUseSky()  # plane surface generation
gen.SetSkySize((0, 0))  # x and y size of the plane
gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center of the plane
gen.SetSeed(1909)

param = 'std'
param = 'guan'
param = 'gaisser'

angle = 'full'
angle = '30deg'

size = '1e4'
size = '1e5'
size = '1e3'
max_mom = '1e5'
max_mom = '1e6'
max_mom = '1e4'

gen.SetMaximumMomentum(int(float(max_mom)))  # in GeV
if angle=='30deg':
    # gen.SetMinimumTheta(np.radians(180-30))  # in degree
    gen.SetMaximumTheta(np.radians(30))  # in degree
if param=='gaisser':
    gen.SetDifferentialFluxGaisser()
if param=='guan':
    gen.SetDifferentialFluxGuan()
file_name = f'EcoMug_{param}_{angle}_{size}_xmom{max_mom}.hdf'
print(f'{file_name}')

t.task('generating arrays')
STATISTICS = int(float(size)) # 1e7:4.5min; 1e6:27s; 2e5:5,4s; 1e4: 0,3s
muon_pos = np.zeros(shape=(STATISTICS, 3), dtype=float)
muon_p = np.zeros(STATISTICS, dtype=float)
muon_theta = np.zeros(STATISTICS, dtype=float)
muon_phi = np.zeros(STATISTICS, dtype=float)
muon_charge = np.zeros(STATISTICS, dtype=int)
muon_e = np.zeros(STATISTICS, dtype=float)

t.task('generating muons')
# t1 = stopwatch.stopwatch(title = 'generating ecomug muons', selfexecutiontime_micros=1.4, time_unit='µs')
for event in tqdm(range(STATISTICS), disable=False):
    # t1.stop(silent=True)
    # t1.task('generate')  # 60% of time
    if param=='std':
        gen.Generate()
    else:
        gen.GenerateFromCustomJ()
    # t1.task('writing data')  # 40% of time
    # t1.task('writing pos')
    muon_pos[event] = gen.GetGenerationPosition()  # 7 µs
    # t1.task('writing p')
    muon_p[event] = gen.GetGenerationMomentum()
    # t1.task('writing theta')
    muon_theta[event] = gen.GetGenerationTheta()
    # t1.task('writing phi')
    muon_phi[event] = gen.GetGenerationPhi()
    # t1.task('writing charge')
    muon_charge[event] = gen.GetCharge()
    # if (event==10):
    #    t1.stop()


t.task('calculation energy')
muon_e = slib.calculate_energy_vectorized_GeV(muon_p)  # faster than for loop

t.task('write to df')
df = pd.DataFrame()
df['pos_x'] = muon_pos[:, 0]
df['pos_y'] = muon_pos[:, 1]
df['pos_z'] = muon_pos[:, 2]
# df['position'] = muon_pos
df['momentum'] = muon_p
df['energy'] = muon_e
df['theta'] = muon_theta
df['phi'] = muon_phi
df['charge'] = muon_charge

t.task('write to HDF file')
# df.to_hdf("data_hdf/"+file_name, key=f'muons_{size}')
df.to_hdf("data_hdf/"+file_name, key=f'main')
t.stop(silent=True)
# print(muon_e)


a = []
t.task('plot data')
for i in muon_e:
    if (i>1000):
        a.append(i)

len_a = len(a)
print(f'myonen mit mehr als 1000 GeV = {len_a}')
if len_a > 100:
    plib.plot_energy_std(
        a, binsize=50,
        xlabel_unit='GeV', show=True)

# quit()
# %%
# GERERATING MULTIPLE single-ENERGYS WITH ECOMUG (fig 4 plot)
############################################################
############################################################
############################################################
############################################################

# 100%|██████████| 5/5 [13:29<00:00, 161.96s/it]
# STATISTICS = int(1e7)
# [1, 10, 50, 100, 1000]
energys = {1: 'darkblue', 10: 'indianred', 50: 'c', 100: 'green', 1000: 'm', 10000: 'r'}
file_name = "EcoMug_muons_2-sphere-r-1.hdf"
file_name = "EcoMug_muons_2-sky-width-10.hdf"
file_name = "EcoMug_gaisser_full.hdf"
file_name = "EcoMug_gaisser_guan_full.hdf"
for gen_momentum in tqdm(energys.keys(), disable=False):
    # continue
    gen = EcoMug.EcoMug()
    gen.SetUseSky()  # plane surface generation
    gen.SetSkySize((10, 10))  # x and y size of the plane
    # gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center
    #   of the plane
    # gen.SetUseHSphere()  # plane Sphere generation
    # gen.SetHSphereRadius(1)
    # gen.SetUseCylinder()
    gen.SetSeed(1909)
    gen.SetDifferentialFluxGuan()
    # gen_momentum = 1
    offset = 0
    gen.SetMinimumMomentum(gen_momentum-offset)  # in GeV
    gen.SetMaximumMomentum(gen_momentum-offset)  # in GeV

    STATISTICS = int(3e6)
    # muon_pos = [None]*STATISTICS
    # muon_p = [float]*STATISTICS
    # muon_theta = [float]*STATISTICS
    # muon_phi = [float]*STATISTICS
    # muon_charge = [float]*STATISTICS

    muon_p = np.zeros(STATISTICS, dtype=float)
    muon_theta = np.zeros(STATISTICS, dtype=float)
    muon_phi = np.zeros(STATISTICS, dtype=float)
    muon_charge = np.zeros(STATISTICS, dtype=int)

    for event in tqdm(range(STATISTICS), disable=False):
        # gen.Generate()
        gen.GenerateFromCustomJ()
        # muon_pos[event] = gen.GetGenerationPosition()
        muon_p[event] = gen.GetGenerationMomentum()
        muon_theta[event] = gen.GetGenerationTheta()
        muon_phi[event] = gen.GetGenerationPhi()
        muon_charge[event] = gen.GetCharge()

    muon_e = slib.calculate_energy_vectorized_GeV(muon_p)

    df = pd.DataFrame()
    # df['position'] = muon_pos
    df['momentum'] = muon_p
    df['energy'] = muon_e
    df['theta'] = muon_theta
    df['phi'] = muon_phi
    df['charge'] = muon_charge
    df.to_hdf(file_name, key=f'GeV{gen_momentum}')
# %%
# %%time
# PLOTTING (FIG. 4)
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
file_name2 = "EcoMug_gaisser_full.hdf"
file_name2 = "EcoMug_gaisser_guan_full.hdf"

binsize = 100

# mu_100gev  = pd.read_hdf(file_name2, key='GeV100')

# plt.rcParams['figure.figsize'] = (8, 8)
# plt.rcParams['font.size'] = 16
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['axes.labelsize'] = 16

for p, c in tqdm(energys.items(), disable=False):
    data = pd.read_hdf(file_name2, key=f'GeV{p}')
    theta = slib.change_zenith_convention(np.array(data['theta']))
    cos_theta = np.cos(theta)
    # theta = change_zenith_convention(np.array(data['theta']))/(np.pi/2)

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
_ = plt.plot(np.cos(x), y,'--', label=r'$\mathrm{cos}^2\theta$')
# _ = plt.plot(x, y,'--', label=r'$\mathrm{cos}^2\theta$')
y = 1/np.cos(x)
_ = plt.plot(np.cos(x), y,'--', label=r'$\frac{1}{\mathrm{cos}\theta}$')

plt.yscale('log')
# plt.xlim([np.pi/2, 0])
# plt.ylim(1e-2, 1e1)
plt.axis([1, 0, 1e-2, 1e1])
plt.xlabel(r'$\mathrm{cos} \;\theta $')
plt.ylabel(r"$I/I_{\mathrm{vert}}$")
plt.title(file_name2)
plt.tight_layout()
plt.legend()
plt.savefig('fig_4_gaisser_guan.pdf')

# %%
data_GeV10_theta0 = np.array(pd.read_hdf("EcoMug_muons2.hdf", key='GeV10')['theta'])
data_GeV1000_theta = np.array(pd.read_hdf("EcoMug_muons2.hdf", key='GeV1000')['theta'])
# %%
# %%time
# theta = np.cos(data['theta'])
# theta = np.cos(data_GeV10_theta)
data_GeV10_theta = slib.change_zenith_convention(data_GeV10_theta0)
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
_ = plt.hist(np.cos(data_GeV10_theta), bins=1000, log=True, density=True, stacked=True) 
plt.tight_layout()

# plt.axis([1, 0, 1e-2, 1e0])
# _ = 
