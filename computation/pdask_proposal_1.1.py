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
FLOAT_TYPE = np.float64
# hdf_folder = 'data_hdf/'
hdf_folder = '/scratch/mschoenfeld/data_hdf/'
show_plots = True
print_results = False
import propagator as proper

# %%
t1 = stopwatch.stopwatch(title='full proposal init and simulation parallized with dask')
t1.task('initialize proposal, making interpol tables')
######################################################################
######################################################################
# def main():

client.upload_file('propagator.py')
reload(proper)
print(f'config : {proper.config}')
reload(stopwatch)
reload(plib)

t1.task('read EcoMug data')
{
}
file_name = "EcoMug_gaisser_30deg_1e4_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e5_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e6_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e7_min5e2_max3e5.hdf"
print(f'{file_name}')
(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        hdf_folder+file_name, f'main')
STATISTICS = len(data_energy)

t1 = stopwatch.stopwatch(title='full simulation: proposal init and simulation')
t1.task('plot EcoMug data')
# plib.plot_energy_std(
#     data_energy, binsize=50,
#     xlabel_unit='GeV', show=show_plots)

t1.task('dask: read hdf, to_bag, map')
partition_setting = slib.which_size(STATISTICS)
# partition_setting = {'npartitions' : 4800}
partition_setting = {'npartitions' : 4800}
partition_setting = {'npartitions' : 480}
partition_setting = {'npartitions' : 48}
partition_setting['npartitions'] = STATISTICS//48
chunksize = partition_setting['npartitions']
chunksize = STATISTICS//48000
# chunksize = 100
with performance_report(filename="dask-report.html"):
    df = dd.read_hdf(hdf_folder+file_name, key='main',
                    columns=['energy', 'theta', 'phi', 'charge'], chunksize=chunksize)
    dfb = df.to_bag()
    # b = df.to_bag()
    # b = client.scatter(b)
    # df = df.persist()
    # Scatter?
    t1.task('bag compute')
    results = dfb.map(proper.pp_propagate) #27s

    # results = b.map(proper.pp_propagate)
    # results = client.map(proper.pp_propagate, df.to_bag(), pure=False)
    # results = client.gather(results)
    # results = client.map(proper.pp_propagate, df, pure=False)
    # results = client.map(proper.pp_propagate, df['energy'], df['theta'], df['phi'], df['charge'], pure = True)
    results = results.compute(pure=False)
    # results = client.gather(results)
    # results = list(results)
    # results.to_hdf(hdf_folder+'results_'+file_name, key=f'main', format='table')
    # results.compute(pure=False)
t1.stop()
#%%
t2 = stopwatch.stopwatch(title='processing of results')
t2.task('nachbereitung')
######################################################################
######################################################################

if (False):
    print('print results')
    print(results)

results_array = np.array(results)

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
df['energies_f'] = energies_f
df['energies_i'] = energies_i
df['distances'] = distances
# df['position'] = muon_pos  # start end points
t2.task('write to HDF file')
file_name = 'results_'+file_name
df.to_hdf(hdf_folder+file_name, key=f'main', format='table')
# dd.compute()

t2.stop()
#%%
t3 = stopwatch.stopwatch(title='plotting of results')
t3.task('print')
######################################################################
######################################################################

# for i in range()
# start_end_points[counter*2] = point1[i]
# start_end_points[counter*2+1] = point2[i]

sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)
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
t3.task('3D plot')
# plib.plot_3D_start_end(
#     start_end_points, proper.detector_pos, detector_size,
#     elev=10, azim=70, alpha=0.3, dpi=1, show=show_plots,
#     title=f'# of particles: {counter}'
# )
# t3.task('distances plot')
# plib.plot_distances_std(
#     distances/100, 100, xlabel_unit='m', show=show_plots
# )
# t3.task('energy plot')
# plib.plot_energy_std(
#     energies_f/1000, binsize=100, xlabel_unit='GeV', show=show_plots, name='E_f at detector'
# )

# # t3.task('energy plot2')
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

t3.stop()

# if __name__ == '__main__':
#     main()
