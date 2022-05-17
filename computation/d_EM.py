# %%
import os, sys
# os.environ['LD_LIBRARY_PATH'] = '/home/mschoenfeld/envs/env_3.9/lib:'
# sys.path.append('/home/mschoenfeld/pkgs/python-3.9.7-h12debd9_1/lib')
# os.chdir(os.path.dirname(__file__))
# print(os.environ['LD_LIBRARY_PATH'])
# print(os.environ['PATH'])
# print(os.environ['PYTHONPATH'])
# sys.path.append('/home/mschoenfeld/envs/env_3.10/lib')
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import proposal as pp

# from EcoMug_pybind11.build import EcoMug
# from numba import vectorize, jit, njit, prange

import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
from importlib import reload

from distributed import Client, LocalCluster, as_completed
import dask.array as da
from dask import delayed
import dask.bag as db
import dask.dataframe as dd
client = Client("localhost:8786") # phobos
# client = Client("tcp://129.217.166.201:8786")
# client = Client("tcp://172.17.79.204:8786")

# hdf_folder = 'data_hdf/'
hdf_folder = '/scratch/mschoenfeld/data_hdf/'

#%
# GERERATING full spectra muons ECOMUG
############################################################
############################################################
############################################################
t = stopwatch.stopwatch(title='generating ecomug muons', selfexecutiontime_micros=0, time_unit='s', return_results=True)
import d_EM_lib as emo
#%
client.upload_file('d_EM_lib.py')
reload(emo)
reload(slib)
reload(plib)
reload(stopwatch)

# file_name = f'Em_{param}_{angle}_{size}_xmom{max_mom}.hdf'
file_name = emo.file_name
print(f'{file_name}')
STATISTICS = int(float(emo.size)) # 1e7:4.5min; 1e6:27s; 2e5:5,4s; 1e4: 0,3s

N_tasks = 23
chunksize = round((STATISTICS/N_tasks))+1

tmphdf = '/tmp/tmp.hdf'
df = pd.DataFrame()
df['0'] = np.zeros(STATISTICS)
df.to_hdf(tmphdf, key=f'main', format='table')

ddf = dd.read_hdf(tmphdf, key='main', columns=['0'], chunksize=chunksize)
b = ddf.to_bag()
# b = db.from_sequence(map(bool, arr), partition_size=chunksize)  # 1/2 slowdown
b = b.map(emo.Ecomug_generate)
b = b.to_dataframe()
t.task('dask compute', True)
results = client.compute(b, pure=False).result()

t.task('post calculations')
#%
results_array = np.array(results)
muon_pos = np.zeros(shape=(STATISTICS, 3))
# muon_pos = np.array(list(results_array[:, 0]))
muon_p = np.array(results_array[:, 1], dtype=float)
muon_theta = np.array(results_array[:, 2], dtype=float)
muon_phi = np.array(results_array[:, 3], dtype=float)
muon_charge = np.array(results_array[:, 4], dtype=np.int8)
muon_e = slib.calculate_energy_vectorized_GeV(muon_p)  # faster than for loop

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
t.task('write to HDF')
df.to_hdf(hdf_folder+file_name, key=f'main', format='table')

elapsed_time_total = t.stop(True)['TOTAL']
# muonen_1e7_time = ( (elapsed_time_total/STATISTICS)*int(1e7))/(60*60)
# print(f'bei aktuellem xmom würden 1e7 muonen {muonen_1e7_time:2.1f} Stunden dauern')
print(f'it took {elapsed_time_total/(60):2.1f} min to complete')

# quit()
#%%
#######################
reload(plib)
plib.plot_energy_std(
    muon_e, binsize=50,
    xlabel_unit='GeV', show=True,
        name='1e7 muons')

a = []
t.task('plot data')
for i in muon_e:
    if (i>800):
        a.append(i)

len_a = len(a)
print(f'myonen mit mehr als 1000 GeV = {len_a} ({len_a/STATISTICS*100:.2f}%)')
if len_a > 100:
    plib.plot_energy_std(
        a, binsize=50,
        xlabel_unit='GeV', show=True)

#%%
file_name = 'Em_gaisser_30deg_1e5_xmom4e5.hdf'
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
    binsize=50,
    show_or_multiplot=False,
    savefig=False,
    histtype='step'
)