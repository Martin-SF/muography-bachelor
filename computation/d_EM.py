# %%
import os, sys
os.chdir(os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import proposal as pp
from importlib import reload

import py_library.my_plots_library as plib
import py_library.simulate_lib as slib
import py_library.stopwatch as stopwatch
import config as config_file

from distributed import Client, LocalCluster
# import dask.bag as db
import dask.dataframe as dd
client = Client("localhost:8786")  # local client


#%
# generating Muons with Ecomug into HDF
############################################################
############################################################
############################################################
t = stopwatch.stopwatch(title='generating ecomug muons', selfexecutiontime_micros=0, time_unit='s', return_results=True)
import d_EM_lib as emo
# client.upload_file('config.py')
client.upload_file('d_EM_lib.py')
reload(emo)
reload(slib)
reload(plib)
reload(stopwatch)
reload(config_file)

print(f'generating {config_file.file_name}')
N_tasks = config_file.N_tasks
chunksize = round((config_file.STATISTICS/N_tasks))+1

# local variable!!!
# workaround für viele tasks 
tmphdf = config_file.hdf_folder+'tmp.hdf'
df = pd.DataFrame()
df['0'] = np.zeros(config_file.STATISTICS)
df.to_hdf(tmphdf, key=f'main', format='table')

ddf = dd.read_hdf(tmphdf, key='main', columns=['0'], chunksize=chunksize)
b = ddf.to_bag()
# b = db.from_sequence(map(bool, arr), partition_size=chunksize)  # 1/2 slowdown
b = b.map(emo.Ecomug_generate)
b = b.to_dataframe()
t.task('calculate muons with EcoMug', True)
results = client.compute(b, pure=False).result()
os.remove(tmphdf)

t.task('post calculations')
results_array = np.array(results)
muon_pos = np.zeros(shape=(config_file.STATISTICS, 3))
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
df.to_hdf(config_file.hdf_folder+config_file.file_name, key=f'main', format='table')

elapsed_time_total = t.stop(True)['TOTAL']
# muonen_1e7_time = ( (elapsed_time_total/config_file.STATISTICS)*int(1e7))/(60*60)
# print(f'bei aktuellem xmom würden 1e7 muonen {muonen_1e7_time:2.1f} Stunden dauern')
print(f'Total time: {elapsed_time_total/(60):2.1f} min')

quit()
#%%
#######################
reload(plib)

plib.plot_hist(
    muon_e, 
    name='EcoMug: energy distribution',
    ylabel = '# of muons',
    xlabel1 = '',
    xlabel2 = 'GeV',
    xlog=True,
    binsize=50,
    show_or_multiplot=config_file.show_or_multiplot,
    savefig=False
)

#%%
energy_threshold = 600  # GeV
a = []
t.task('plot data')
for i in muon_e:
    if (i>energy_threshold):
        a.append(i)

len_a = len(a)
print(f'myonen mit mehr als {energy_threshold} GeV = {len_a} ({len_a/config_file.STATISTICS*100:.2f}%)')
if len_a > 100:
    plib.plot_hist(
        a, 
        name=f'EcoMug: energy distribution of E>{energy_threshold} GeV',
        xlabel1 = 'E',
        xlabel2 = 'GeV',
        show_or_multiplot=config_file.show_or_multiplot,
        xlog=True
    )


#%%
reload(plib)
# config_file.file_name = 'Em_gaisser_30deg_1e5_xmom4e5.hdf'
# (data_position, data_momentum, data_energy,
#     data_theta, data_phi, data_charge) = slib.read_muon_data(
#         config_file.hdf_folder+config_file.file_name, f'main')

plib.plot_hist(
    np.degrees(muon_theta), 
    name='EcoMug: theta distribution',
    ylabel = '# of muons',
    xlabel1 = r'\theta',
    xlabel2 = '°',
    label=r'$\theta \;\;all$',
    xlog=False,
    binsize=50,
    show_or_multiplot=config_file.show_or_multiplot,
    savefig=True
)