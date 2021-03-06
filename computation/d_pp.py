# %%
# print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os, sys
from uncertainties import ufloat
# from numba import jit, njit, vectorize, prange
from importlib import reload
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
import proposal as pp
from distributed import Client, LocalCluster
from dask.distributed import performance_report
import dask.dataframe as dd
import dask.bag as db
import config as config_file


plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14
plt.rcParams.update({'figure.dpi':70})
# os.chdir(os.path.dirname(__file__))  # wichtig wenn nicht über ipython ausgeführt
client = Client("localhost:8786") # phobos
FLOAT_TYPE = np.float64


print(f'{config_file.file_name} | N_tasks = {config_file.N_tasks}')
# %
######################################################################
######################################################################
t1 = stopwatch.stopwatch(
    title='full proposal init and simulation parallized with dask',
    time_unit='s')
t1.task('initialize proposal, making interpol tables')

import d_pp_lib as proper
client.upload_file('d_pp_lib.py')
client.upload_file('config.py')
reload(proper)
reload(stopwatch)
reload(plib)
reload(config_file)
print(f'PROPOSAL config = {config_file.PP_config}, {config_file.pp_config_string}')

STATISTICS = len(pd.read_hdf(
    config_file.hdf_folder+config_file.file_name, key=f'main', columns=['charge']))
chunksize = round((STATISTICS/config_file.N_tasks))+1

meta={
    'hit_detector': bool, 
    'distances_f_raw': FLOAT_TYPE, 
    'energies_f_raw': FLOAT_TYPE, 
    'energies_i_raw': FLOAT_TYPE, 
    'point1x_raw': FLOAT_TYPE, 
    'point1y_raw': FLOAT_TYPE, 
    'point1z_raw': FLOAT_TYPE, 
    'point2x_raw': FLOAT_TYPE,
    'point2y_raw': FLOAT_TYPE,
    'point2z_raw': FLOAT_TYPE
}

t1.task('propagating', True)
''' 
future = client.submit(func, big_data)    # bad

    big_future = client.scatter(big_data)     # good
    future = client.submit(func, big_future)  # good
    # own approach
    ddf = client.scatter(ddf)
    dfb = ddf.result().to_bag()
    # dfb = client.map(proper.pp_propagate, db.from_delayed(dfb) ) #27s
    # dfb = client.map(proper.pp_propagate, dfb) #27s
'''
with performance_report(filename="dask-report.html"):
    ddf = dd.read_hdf(config_file.hdf_folder+config_file.file_name, key='main',
                    columns=['energy', 'theta', 'phi', 'charge'], chunksize=chunksize)
    dfb = ddf.to_bag()

    dfb = dfb.map(proper.pp_propagate) #27s
    ddfr = dfb.to_dataframe(meta=meta)
    # ddfr.to_hdf(config_file.hdf_folder+'results_raw_'+config_file.file_name, key=f'main', format='table')
    results = client.compute(ddfr, pure=False).result()

t1.stop(config_file.silent)
# %
t2 = stopwatch.stopwatch(title='processing of results')
t2.task('nachbereitung')
# todo die nachbereitung mit dask arrays beschleunigen
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
#%%
t2.task('write to HDF file')
df.to_hdf(config_file.hdf_folder+config_file.file_name_results, key=f'main', format='table')

s1 = f'({counter:.1f}) of {STATISTICS:.0e} ({counter/STATISTICS*100:.4})% detector hits'
# counter_u = ufloat(counter, np.sqrt(counter))
# s1 = f'({counter_u:.1f}) of {STATISTICS:.0e} ({counter_u/STATISTICS*100:.4})% detector hits'
s2 = f'min(E_i) at detector = {min(energies_i)/1000:.1f} GeV'
print(f'{s1} | {s2}')

t2.stop(config_file.silent)
quit()
# %%
# t3 = stopwatch.stopwatch(title='plotting of results')
# plots
######################################################################
######################################################################
######################################################################
######################################################################
reload(plib)
# t3.task('3D plot')


#%

# config_file.file_name = "EcoMug_gaisser_30deg_1e6_min5e2_max3e5.hdf"
# df = pd.read_hdf(config_file.hdf_folder+'results_'+config_file.file_name, key='main')

# energies_f = df['energies_f']
# energies_i = df['energies_i'] 
# distances = df['distances']

plib.plot_3D_start_end(
    start_end_points/100,
    elev=15, azim=70, alpha=0.14, dpi=1, show=config_file.show_plots,
    title=f'# of particles: {counter}'
)


# depreciated: original thought: plot a cubic detector
sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)

# plib.plot_3D_start_end(
#     start_end_points, proper.detector_pos, detector_size,
#     elev=10, azim=70, alpha=0.3, dpi=1, show=config_file.show_plots,
#     title=f'# of particles: {counter}'
# )
#%%
# t3.task('distances plot')
# plib.plot_distances_std(
#     distances/100, 100, xlabel2='m', show=config_file.show_plots
# )
# t3.task('energy plot')
# plib.plot_energy_std(
#     energies_f/1000, binsize=100, xlabel2='GeV', show=config_file.show_plots, name='E_f at detector'
# )

# # t3.task('energy plot2')
# plib.plot_energy_std(
#     energies_i/1000, binsize=100, xlabel2='GeV', show=config_file.show_plots, name='E_i at Detector'
# )

plib.plot_hist(
    energies_i/1000, 
    ylabel = '# of muons',
    xlabel1 = 'E',
    xlabel2 = 'GeV',
    label=r'$E_i$',
    xlog=True,
    binsize=100,
    show_or_multiplot='multi',
    histtype='step'
)
plib.plot_hist(
    energies_f/1000, 
    name='Muons at detector',
    ylabel = '# of muons',
    xlabel1 = 'E',
    xlabel2 = 'GeV',
    label=r'$E_f$',
    xlog=True,
    binsize=100,
    show_or_multiplot=config_file.show_or_multiplot,
    savefig=True,
    histtype='step'
)
plib.plot_hist(
    end_points[:, 2]/100*(-1),
    ylabel = '# of muons',
    xlabel1 = 'd',
    xlabel2 = 'm',
    label=r'$muons$',
    name='z-coordinate of muons at detector',
    xlog=True,
    binsize=10,
    show_or_multiplot=config_file.show_or_multiplot,
    savefig=True,
    histtype='bar'
)
plib.plot_hist(
    distances_f/100/1000, 
    name='total distance of muons at detector',
    ylabel = '# of muons',
    xlabel1 = 'd',
    xlabel2 = 'm',
    label=r'$muons$',
    xlog=False,
    binsize=70,
    show_or_multiplot=config_file.show_or_multiplot,
    savefig=True,
    histtype='bar'
)


# t3.stop(config_file.silent)

# if __name__ == '__main__':
#     main()

# %%
