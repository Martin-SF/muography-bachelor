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
# from numba import vectorize
# from numba import jit, njit, prange
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
from importlib import reload
reload(slib)
reload(plib)
reload(stopwatch)


from distributed import Client, LocalCluster, as_completed
#%%
# client = Client("localhost:8786")
client = Client("localhost:8786") # phobos
client = Client("tcp://localhost:8786") # phobos

# client.upload_file('EcoMug_pybind11/build/EcoMug.cpython-39-x86_64-linux-gnu.so')
client.upload_file('EcoMug_objekt_dask.py')
# client = Client("tcp://129.217.166.201:8786")
# client = Client("tcp://172.17.79.204:8786")
# client = Client("localhost:43887")
# client = Client()

# slib.change_zenith_convention(0)
# slib.calculate_energy_vectorized_GeV(0)

# GERERATING full spectra muons ECOMUG
############################################################
############################################################
############################################################
############################################################
t = stopwatch.stopwatch(title='generating ecomug muons', selfexecutiontime_micros=0, time_unit='ms')
t.task('generating ecomug object')
import EcoMug_objekt_dask as emo

file_name = emo.file_name
print(f'{file_name}')

t.task('generating arrays')
STATISTICS = int(float(emo.size)) # 1e7:4.5min; 1e6:27s; 2e5:5,4s; 1e4: 0,3s
muon_pos = np.zeros(shape=(STATISTICS, 3), dtype=float)
muon_p = np.zeros(STATISTICS, dtype=float)
muon_theta = np.zeros(STATISTICS, dtype=float)
muon_phi = np.zeros(STATISTICS, dtype=float)
muon_charge = np.zeros(STATISTICS, dtype=int)
muon_e = np.zeros(STATISTICS, dtype=float)

t.task('generating muons')
# t1 = stopwatch.stopwatch(title = 'generating ecomug muons', selfexecutiontime_micros=1.4, time_unit='µs')

def Ecomug_generate(i):
    # t1.stop(silent=True)
    # t1.task('generate')  # 60% of time
    # emo.genGenerate()gen.GenerateFromCustomJ()
    emo.gen.GenerateFromCustomJ()
    muon_pos = emo.gen.GetGenerationPosition()  # 7 µs
    muon_p = emo.gen.GetGenerationMomentum()
    muon_theta = emo.gen.GetGenerationTheta()
    muon_phi = emo.gen.GetGenerationPhi()
    muon_charge = emo.gen.GetCharge()
    return (muon_pos, muon_p, muon_theta, muon_phi, muon_charge)

# %%
# STATISTICS = int(1e3)

futures = client.map(Ecomug_generate, range(STATISTICS), pure = True)
results = client.gather(futures)

# for event in tqdm(range(STATISTICS), disable=False):
#     future = client.submit(Ecomug_generate)
# results = future.result()

print_results = False
if (print_results):    
        print('print results')
        print(results)
t.stop()

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
