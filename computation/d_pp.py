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
plt.rcParams.update({'figure.dpi':70})
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
worker_number = 24
{
'''
1e7
workers:
24 -> crash coudnlt gather keys
2400 crash 
240 crash 
'''
}
show_plots = True
print_results = False
silent = True
file_name = "EcoMug_gaisser_30deg_1e4_min6e2_max2e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e5_min6e2_max2e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e6_min6e2_max2e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e5_min6e2_max2e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e6_min4e2_max2e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e7_min2e2_max2e5.hdf"
print(f'{file_name}')

# %
######################################################################
######################################################################
t1 = stopwatch.stopwatch(
    title='full proposal init and simulation parallized with dask',
    time_unit='s')
t1.task('initialize proposal, making interpol tables')

import d_pp_lib as proper
client.upload_file('d_pp_lib.py')
reload(proper)
print(f'config : {proper.config}')
reload(stopwatch)
reload(plib)

STATISTICS = len(pd.read_hdf(
    hdf_folder+file_name, key=f'main', columns=['charge']))
chunksize = round((STATISTICS/worker_number))+1

meta={'hit_detector': bool, 
    'distances_f_raw': FLOAT_TYPE, 
    'energies_f_raw': FLOAT_TYPE, 
    'energies_i_raw': FLOAT_TYPE, 
    'point1x_raw': FLOAT_TYPE, 
    'point1y_raw': FLOAT_TYPE, 
    'point1z_raw': FLOAT_TYPE, 
    'point2x_raw': FLOAT_TYPE,
    'point2y_raw': FLOAT_TYPE,
    'point2z_raw': FLOAT_TYPE}

t1.task('dask compute', True)
with performance_report(filename="dask-report.html"):
    ddf = dd.read_hdf(hdf_folder+file_name, key='main',
                    columns=['energy', 'theta', 'phi', 'charge'], chunksize=chunksize)
    dfb = ddf.to_bag()
    dfb = dfb.map(proper.pp_propagate) #27s
    ddfr = dfb.to_dataframe(meta=meta)
    # ddfr.to_hdf(hdf_folder+'results_raw_'+file_name, key=f'main', format='table')
    results = client.compute(ddfr).result()

t1.stop(silent)
# %
t2 = stopwatch.stopwatch(title='processing of results')
t2.task('nachbereitung')
######################################################################
######################################################################


hit_detector = np.array(results['hit_detector'], dtype=bool)
distances_f_raw = np.array(results['distances_f_raw'], dtype=FLOAT_TYPE)
energies_f_raw = np.array(results['energies_f_raw'], dtype=FLOAT_TYPE)
energies_i_raw = np.array(results['energies_i_raw'], dtype=FLOAT_TYPE)
point1x_raw = np.array(results['point1x_raw'], dtype=FLOAT_TYPE)
point1y_raw = np.array(results['point1y_raw'], dtype=FLOAT_TYPE)
point1z_raw = np.array(results['point1z_raw'], dtype=FLOAT_TYPE)
point2x_raw = np.array(results['point2x_raw'], dtype=FLOAT_TYPE)
point2y_raw = np.array(results['point2y_raw'], dtype=FLOAT_TYPE)
point2z_raw = np.array(results['point2z_raw'], dtype=FLOAT_TYPE)

counter = int(sum(hit_detector))  #len von allen die True sind
energies_f = np.zeros(counter, dtype=FLOAT_TYPE)
energies_i = np.zeros(counter, dtype=FLOAT_TYPE)
distances_f = np.zeros(counter, dtype=FLOAT_TYPE)
start_points = np.zeros(shape=(counter, 3), dtype=FLOAT_TYPE)
end_points = np.zeros(shape=(counter, 3), dtype=FLOAT_TYPE)
start_end_points = np.zeros(shape=(counter*2, 3), dtype=FLOAT_TYPE)

t2.task('1')
i2 = 0
for i in range(STATISTICS):
    if hit_detector[i] == True:
        energies_f[i2] = energies_f_raw[i]
        energies_i[i2] = energies_i_raw[i]
        distances_f[i2] = distances_f_raw[i]

        start_points[i2] = np.array([point1x_raw[i], point1y_raw[i], point1z_raw[i]])
        end_points[i2] = np.array([point2x_raw[i], point2y_raw[i], point2z_raw[i]])

        start_end_points[i2*2] = start_points[i2]
        start_end_points[i2*2+1] = end_points[i2]
        i2 += 1

t2.task('2')
df = pd.DataFrame()
df['energies_f'] = energies_f
df['energies_i'] = energies_i
df['distances'] = distances_f
df['point1x'] = start_points[:, 0]
df['point1y'] = start_points[:, 1]
df['point1z'] = start_points[:, 2]
df['point2x'] = end_points[:, 0]
df['point2y'] = end_points[:, 1]
df['point2z'] = end_points[:, 2]

t2.task('3')
t2.task('write to HDF file')
df.to_hdf(hdf_folder+'results_'+file_name, key=f'main', format='table')

t2.task('4')
print(
f'{counter} of {STATISTICS} muons ({counter/STATISTICS*100:.4}%) ' +
'hit the detector'
)
print(f'min(E_i) at detector is {min(energies_i)/1000:.1f} GeV')

t2.stop(silent)
# %%
t3 = stopwatch.stopwatch(title='plotting of results')
# plots
######################################################################
######################################################################
######################################################################
######################################################################
reload(plib)
t3.task('3D plot')

sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)


#%

# file_name = "EcoMug_gaisser_30deg_1e6_min5e2_max3e5.hdf"
# df = pd.read_hdf(hdf_folder+'results_'+file_name, key='main')

# energies_f = df['energies_f']
# energies_i = df['energies_i'] 
# distances = df['distances']

plib.plot_3D_start_end(
    start_end_points/100,
    elev=10, azim=70, alpha=0.3, dpi=1, show=show_plots,
    title=f'# of particles: {counter}'
)
# plib.plot_3D_start_end(
#     start_end_points, proper.detector_pos, detector_size,
#     elev=10, azim=70, alpha=0.3, dpi=1, show=show_plots,
#     title=f'# of particles: {counter}'
# )
#%%
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

plib.plot_hist(
    energies_i/1000, 
    ylabel = '# of muons',
    x_label1 = 'E',
    xlabel_unit = 'GeV',
    label=r'$E_i$',
    xlog=True,
    binsize=100,
    show_and_nomultiplot=False,
    histtype='step'
)
plib.plot_hist(
    energies_f/1000, 
    name='Muons at detector',
    ylabel = '# of muons',
    x_label1 = 'E',
    xlabel_unit = 'GeV',
    label=r'$E_f$',
    xlog=True,
    binsize=100,
    show_and_nomultiplot=True,
    savefig=True,
    histtype='step'
)
plib.plot_hist(
    end_points[:, 2]/100*(-1),
    ylabel = '# of muons',
    x_label1 = 'd',
    xlabel_unit = 'm',
    label=r'$muons$',
    name='z-coordinate of muons at detector',
    xlog=True,
    binsize=100,
    show_and_nomultiplot=True,
    savefig=True,
    histtype='bar'
)
plib.plot_hist(
    distances/100/1000, 
    name='total distance of muons at detector',
    ylabel = '# of muons',
    x_label1 = 'd',
    xlabel_unit = 'm',
    label=r'$muons$',
    xlog=False,
    binsize=30,
    show_and_nomultiplot=True,
    savefig=True,
    histtype='bar'
)


t3.stop(silent)

# if __name__ == '__main__':
#     main()

# %%
