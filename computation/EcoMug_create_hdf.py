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
from importlib import reload
reload(slib)
reload(plib)
reload(stopwatch)

# slib.change_zenith_convention(0)
# slib.calculate_energy_vectorized_GeV(0)

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
size = '1e4'
max_mom = '1e5'
max_mom = '1e6'
max_mom = '1e4'
# 01:17


gen.SetMaximumMomentum(int(float(max_mom)))  # in GeV
if angle=='30deg':
    gen.SetMaximumTheta(np.radians(30))  # in degree
if param=='gaisser':
    gen.SetDifferentialFluxGaisser()
elif param=='guan':
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
hdf_folder = '/scratch/mschoenfeld/data_hdf/'
# df.to_hdf(hdf_folder+file_name, key=f'muons_{size}')
df.to_hdf(hdf_folder+file_name, key=f'main')
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

data_theta = muon_theta
file_name = 'EcoMug_std_full_1e4_xmom1e6.hdf'
(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        hdf_folder+file_name, f'main')
plib.plot_hist(
    np.degrees(data_theta), 
    ylabel = '# of muons',
    x_label1 = '\theta',
    xlabel_unit = '°',
    label=r'$\theta \;\;all$',
    xlog=False,
    binsize=30,
    show_or_multiplot=False,
    savefig=True,
    histtype='step'
)


file_name = 'EcoMug_std_30deg_1e4_xmom1e6.hdf'
(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        hdf_folder+file_name, f'main')
plib.plot_hist(
    np.degrees(data_theta), 
    ylabel='# of muons',
    x_label1=r'\theta',
    xlabel_unit='°',
    name='',
    label=r'$\theta < 30°$',
    xlog=False,
    binsize=20,
    show_or_multiplot=True,
    savefig=True,
    histtype='step'
)
