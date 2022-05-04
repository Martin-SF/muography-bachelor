# %%
# print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os, sys
# os.chdir(os.path.dirname(__file__))  # wichtig wenn nicht über ipython ausgeführt

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14
# from numba import jit, njit, vectorize, prange
from importlib import reload
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
import proposal as pp
from distributed import Client
from dask.distributed import performance_report
import dask.dataframe as dd
import dask.bag as db
client = Client("localhost:8786") # phobos
t = stopwatch.stopwatch()
FLOAT_TYPE = np.float64
# hdf_folder = 'data_hdf/'
hdf_folder = '/scratch/mschoenfeld/data_hdf/'
show_plots = True


#%%
# start
######################################################################
######################################################################
######################################################################
# show_plots = False
print_results = False
t.settings(title='full proposal propagation and plotting')

t.task('read EcoMug data')
file_name = "EcoMug_gaisser_30deg_1e5_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e6_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e7_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e4_min5e2_max3e5.hdf"
print(f'{file_name}')
# size = int(1e7)
# b = db.from_sequence([hdf_folder+file_name]).map(slib.read_muon_data, f'main')
# b = b.map(proper.pp_propagate)
# results = b.compute(pure=False)

(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        hdf_folder+file_name, f'main')
STATISTICS = len(data_energy)
t.task('plot data')
plib.plot_energy_std(
    data_energy, binsize=50,
    xlabel_unit='GeV', show=show_plots)
# t.stop()
# %%
# proposal-propagation
######################################################################
######################################################################
######################################################################
# reload(stopwatch)
# reload(plib)
t.task('initilaize propagation')
sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)

t.task('propagation')
# def main():

client.upload_file('propagator.py')
import propagator as proper
reload(proper)

#

def which_size(a):
    if a<=int(1e4):
        return {'npartitions' : 100}
    if a<=int(1e5):
        return {'npartitions' : 500}
    if a<=int(1e6):
        return {'npartitions' : 1000}
    if a<=int(1e7) or a>=int(1e7):
        return {'npartitions' : 10000}
partition_setting = which_size(STATISTICS)
# partition_setting = {'npartitions' : 4800}
partition_setting = {'npartitions' : 4800}
partition_setting = {'npartitions' : 480}

t.task('bags erstellen')
# b_position = db.from_sequence(data_position, **partition_setting)



    # b_energy = db.from_sequence(da.from_array(data_energy, chunks='100 MiB'), **partition_setting)
    # b_theta = db.from_sequence(da.from_array(data_theta, chunks='100 MiB'), **partition_setting)
    # b_phi = db.from_sequence(da.from_array(data_phi, chunks='100 MiB'), **partition_setting)
    # b_charge = db.from_sequence(da.from_array(data_charge, chunks='100 MiB'), **partition_setting)

# b_energy = db.from_sequence(data_energy, **partition_setting)
# b_theta = db.from_sequence(data_theta, **partition_setting)
# b_phi = db.from_sequence(data_phi, **partition_setting)
# b_charge = db.from_sequence(data_charge, **partition_setting)


df = dd.read_hdf(hdf_folder+file_name, key='main',
                columns=['energy', 'theta', 'phi', 'charge'], chunksize=partition_setting['npartitions'])
#%
df = df.to_bag()
# df = df.persist()
# df = df.repartition(**partition_setting)
# b_energy = db.from_sequence(df['energy'], **partition_setting)
# b_theta = db.from_sequence(df['theta'], **partition_setting)
# b_phi = db.from_sequence(df['phi'], **partition_setting)
# b_charge = db.from_sequence(df['charge'], **partition_setting)



with performance_report(filename="dask-report.html"):
    ## some dask computation
    # b = client.map(proper.pp_propagate, data_energy, data_theta, data_phi, data_charge)
    t.task('bags compute')
    results = df.map(proper.pp_propagate).compute(pure=False)
    # results = b_energy.map(proper.pp_propagate, b_theta, b_phi, b_charge).compute(pure=False)

    # results = client.compute(b, pure=False)
    # futures = client.map(proper.pp_propagate, df['energy'], df['theta'], df['phi'], df['charge'], pure = True)
    # results = client.gather(futures)
t.stop()
#%%
# results = client.gather(b, pure=False)

# results   = b.persist(pure=False)
# b = client.persist(b)
# results = client.persist(b, pure=False)
# results = client.gather(results)
# Client.scatter

t.task('nachbereitung')
results_array = np.array(results)

# (hit_detector, distance_at_track_end, energy_at_track_end, 
#   energy_init, point1, point2)
if (False):    
    print('print results')
    print(results)
# t.stop()

hit_detector = np.array(results_array[:, 0], dtype=bool)
distances_f_raw = np.array(results_array[:, 1], dtype=FLOAT_TYPE)
energies_f_raw = np.array(results_array[:, 2], dtype=FLOAT_TYPE)
energies_i_raw = np.array(results_array[:, 3], dtype=FLOAT_TYPE)
# point1_raw = np.array(results_array[:, 4], dtype=float)
# point2_raw = np.array(results_array[:, 5], dtype=float)
counter = int(sum(hit_detector))  #len von allen die True sind
distances = np.zeros(counter, dtype=FLOAT_TYPE)
energies_f = np.zeros(counter, dtype=FLOAT_TYPE)
energies_i = np.zeros(counter, dtype=FLOAT_TYPE)
# start_end_points = np.zeros(shape=(STATISTICS*2, 3), dtype=FLOAT_TYPE)
# point1 = []
# point2 = []
i2 = 0
for i in range(len(hit_detector)):
    if hit_detector[i] == True:
        energies_f[i2] = energies_f_raw[i]
        energies_i[i2] = energies_i_raw[i]
        distances[i2] = distances_f_raw[i]
        i2 += 1


df = pd.DataFrame()
# df['hit_detector'] = hit_detector
# df['position'] = muon_pos
df['energies_f'] = energies_f
df['energies_i'] = energies_i
df['distances'] = distances
t.task('write to HDF file')
file_name = 'results'+file_name
df.to_hdf(hdf_folder+file_name, key=f'main', format='table')

t.stop()



#%%
# for i in range()
# start_end_points[counter*2] = point1[i]
# start_end_points[counter*2+1] = point2[i]


print(
f'{counter} of {STATISTICS} muons ({counter/STATISTICS*100:.4}%) ' +
'hit the detector'
)
print(f'min E_i at detector is {min(energies_i)/1000:.1f} GeV')

# %
# plots
######################################################################
######################################################################
######################################################################
######################################################################
reload(plib)
t.settings(title=None)
t.task('3D plot')
# plib.plot_3D_start_end(
#     start_end_points, proper.detector_pos, detector_size,
#     elev=10, azim=70, alpha=0.3, dpi=1, show=show_plots,
#     title=f'# of particles: {counter}'
# )
# t.task('distances plot')
# plib.plot_distances_std(
#     distances/100, 100, xlabel_unit='m', show=show_plots
# )
# t.task('energy plot')
# plib.plot_energy_std(
#     energies_f/1000, binsize=100, xlabel_unit='GeV', show=show_plots, name='E_f at detector'
# )

# # t.task('energy plot2')
# plib.plot_energy_std(
#     energies_i/1000, binsize=100, xlabel_unit='GeV', show=show_plots, name='E_i at Detector'
# )

# plib.plot_hist(
#     energies_i/1000, 
#     ylabel = '# of muons',
#     x_label1 = 'E',
#     xlabel_unit = 'GeV',
#     label=r'$muons$',
#     xlog=True,
#     binsize=30,
#     show_or_multiplot=False,
#     savefig=True,
#     histtype='bar'
# )
plib.plot_hist(
    energies_f/1000, 
    ylabel = '# of muons',
    x_label1 = 'E',
    xlabel_unit = 'GeV',
    label=r'$muons$',
    xlog=True,
    binsize=30,
    show_or_multiplot=False,
    savefig=True,
    histtype='bar'
)

t.stop(silent=True)

# if __name__ == '__main__':
#     main()

# %%
