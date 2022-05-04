# %%
# from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy.random as random 
# from scipy.stats import norm
# import os, sys
import proposal as pp
from EcoMug_pybind11.build import EcoMug
from numba import vectorize
# from numba import jit, njit, prange
from importlib import reload
import py_library.stopwatch as stopwatch
import py_library.my_plots_library as plib


MU_MINUS_MASS = pp.particle.MuMinusDef().mass


@vectorize(nopython=True)
def change_azimuth_convention(angle_in_rad):
    return -angle_in_rad + np.pi


# calculate energy from momentum, expecting GeV, 
# calculating MU_MINUS_MASS to GeV with One_momentum_in_MeV
@vectorize(nopython=True)
def calculate_energy_vectorized_GeV(momentum):
    One_momentum_in_MeV = 1000
    return np.sqrt(momentum * momentum +
                        (MU_MINUS_MASS/One_momentum_in_MeV)**2)


change_azimuth_convention(0)
calculate_energy_vectorized_GeV(0)


# os.chdir(os.path.dirname(sys.argv[0]))

# %%
# %%time
# GERERATING spectras WITH ECOMUG
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
gen.SetDifferentialFluxGuan()
# gen.SetMinimumMomentum(60)  # in GeV
# 66 for min 100 standardrock

gen.SetSeed(1909)
STATISTICS = int(2e4) # 1e7:4.5min; 1e6:27s; 2e5:5,4s; 1e4: 0,3s
STATISTICS = int(1e5) # 1e7:4.5min; 1e6:27s; 2e5:5,4s; 1e4: 0,3s

t.task('generating arrays')
muon_pos = np.zeros(shape=(STATISTICS, 3), dtype=float)
muon_p = np.zeros(STATISTICS, dtype=float)
# muon_p2 = [float]*STATISTICS
muon_p2 = []
muon_theta = np.zeros(STATISTICS, dtype=float)
muon_phi = np.zeros(STATISTICS, dtype=float)
muon_charge = np.zeros(STATISTICS, dtype=int)
muon_e = np.zeros(STATISTICS, dtype=float)

def wsk(p):
    return 0.9/(1 + np.exp(((-1)*p/100)+5))+0.1  # meine angenäherte fkt

weighted_counter = 0
t.task('generating muons')
# t1 = stopwatch.stopwatch(title = 'generating ecomug muons', selfexecutiontime_in_ms=1.4, time_unit='µs')
for event in tqdm(range(STATISTICS), disable=False):
    # t1.stop(silent=True)
    # t1.task('generate')  # 60% of time
    # gen.Generate()
    gen.GenerateFromCustomJ()
    # t1.task('writing data')  # 40% of time
    muon_p[event] = gen.GetGenerationMomentum()
    p = muon_p[event]
    
    if np.random.random() <= wsk(p):  # gleiche noch mit distanz
        muon_p2.append(p)
        # continue
    #     # das weighting in ein extra array um das beim histogrammieren später berücksichtigen zu können.
    #     # weighted_counter += 1

    
    # detector_counter += 1/wsk(p)

    # t1.task('writing pos')
    # muon_pos[event] = gen.GetGenerationPosition()  # 7 µs
    # # t1.task('writing p')
    # # t1.task('writing theta')
    # muon_theta[event] = gen.GetGenerationTheta()
    # # t1.task('writing phi')
    # muon_phi[event] = gen.GetGenerationPhi()
    # # t1.task('writing charge')
    # muon_charge[event] = gen.GetCharge()
    # if (event==STATISTICS-1):
    #    t1.stop()

#%%

# reload(plib)
plib.plot_energy_std(
    muon_p2, binsize=40, xlabel_unit='GeV', show=True, histtype='step', name='ungewichtet'
)

plib.plot_energy_std(
    muon_p, binsize=40, xlabel_unit='GeV', show=True, histtype='step', name='all'
)

plib.plot_energy_std(
    muon_p2, binsize=40, xlabel_unit='GeV', show=True, histtype='step', weights=[1/wsk(p) for p in muon_p2], name='gewichtet'
)
plt.legend()
#%%
x = np.geomspace(1e-2, 1e3)
x = np.geomspace(1, 20)
y = lambda x: 1/(1+np.exp(-(x-10)))
plt.plot(x, y(x))
#%%
x = np.geomspace(1e-2, 1e3)
x = np.geomspace(1, 1e3)
def E_lim_function(x):
    return 1/(1+np.exp(-(x-10)))
def y(x, E_lim, STATISTICS):
    if (x < E_lim):
        return E_lim_function(x)
    else:
        m*x+E_lim_function(E_lim)-m*Elim

# y = lambda x: 1/(1+np.exp(-(x-10)))
E_lim = 10
STATISTICS = 1000
plt.plot(x, y(x, E_lim, STATISTICS))
plt.xscale('log')

# %%
t.task('calculation energy')
muon_e = calculate_energy_vectorized_GeV(muon_p)  # faster than for loop

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
df.to_hdf(file_name, key=f'muons_{STATISTICS}')
t.stop(silent=False)
print(muon_e)


# %%
# GERERATING MULTIPLE single-ENERGYS WITH ECOMUG (fig 4 plot)
############################################################
############################################################
############################################################
############################################################

# 100%|██████████| 5/5 [13:29<00:00, 161.96s/it]
# STATISTICS = int(1e7)
# [1, 10, 50, 100, 1000]
energys = {1: 'darkblue', 10: 'indianred', 50: 'c', 100: 'green', 1000: 'm'}
file_name = "EcoMug_muons_2-sphere-r-1.hdf"
file_name = "EcoMug_muons_2-sky-width-10.hdf"
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
        muon_pos[event] = gen.GetGenerationPosition()
        muon_p[event] = gen.GetGenerationMomentum()
        muon_theta[event] = gen.GetGenerationTheta()
        muon_phi[event] = gen.GetGenerationPhi()
        muon_charge[event] = gen.GetCharge()

    muon_e = calculate_energy_vectorized_GeV(muon_p)

    df = pd.DataFrame()
    df['position'] = muon_pos
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

# %%
data_GeV10_theta0 = np.array(pd.read_hdf("EcoMug_muons2.hdf", key='GeV10')['theta'])
data_GeV1000_theta = np.array(pd.read_hdf("EcoMug_muons2.hdf", key='GeV1000')['theta'])
# %%
# %%time
# theta = np.cos(data['theta'])
# theta = np.cos(data_GeV10_theta)
data_GeV10_theta = change_azimuth_convention(data_GeV10_theta0)
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
