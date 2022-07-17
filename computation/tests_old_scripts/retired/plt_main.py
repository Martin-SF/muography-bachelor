#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14
plt.rcParams.update({'figure.dpi':70})
from tqdm import tqdm
import pandas as pd
import os
import config as config_file
from importlib import reload

reload(config_file)
os.chdir(os.path.dirname(__file__))
# from scipy.stats import norm
# import proposal as pp
# from numba import vectorize
# from numba import jit, njit, prange
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
from importlib import reload
reload(slib)
reload(plib)
reload(stopwatch)
show_plots = True
print_results = False
silent = True
FLOAT_TYPE = np.float64

# x = np.geomspace(1, int(1e6))
# def f(x):
#     return np.e**((-x))
#     return np.e**((-1)*np.e**(-x))
# plt.yscale('log')
# # plt.plot(x, f(np.log(x)))
# plt.plot(x, 1/x, scalex=100000)


# %%
# proposal plots
t3 = stopwatch.stopwatch(start=True, title='plotting of results')


# file_name = "EcoMug_gaisser_30deg_1e5_min2e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e6_min2e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e4_min7e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e6_min7e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e6_min6e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e5_min6e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e7_min2e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e7_min6e2_max2e5.hdf"

# file_name = "EcoMug_1e4_gaisser_min6e2_max2e5_30deg.hdf"


(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        config_file.hdf_folder+config_file.file_name, f'main')

df = pd.read_hdf(config_file.hdf_folder+config_file.file_name_results, key='main')
# df = pd.read_hdf(config_file.hdf_folder+config_file.file_name, key='main')

# counter = len(df['point1x'])
# start_points = np.zeros(shape=(counter, 3), dtype=FLOAT_TYPE)
# end_points = np.zeros(shape=(counter, 3), dtype=FLOAT_TYPE)
# start_end_points = np.zeros(shape=(counter*2, 3), dtype=FLOAT_TYPE)

# for i in range(counter):
#     start_points[i] = [df['point1x'][i], df['point1y'][i], df['point1z'][i]]
#     end_points[i] = [df['point2x'][i], df['point2y'][i], df['point2z'][i]]
#     start_end_points[i*2] = start_points[i]
#     start_end_points[i*2+1] = end_points[i]


# energies_f = df['energies_f']
# energies_i = df['energies_i'] 
# distances = df['distances']
#%%
reload(plib)
plib.plot_3D_start_end(
    start_end_points/100,
    elev=10, azim=30, alpha=0.1, dpi=1, show=show_plots,
    title=f'# of particles: {counter}', name='pp_3D_start_end'
)
# t3.task('distances plot')
# plib.plot_distances_std(
#     distances/100, 100, xlabel2='m', show=show_plots
# )
# t3.task('energy plot')
# plib.plot_energy_std(
#     energies_f/1000, binsize=100, xlabel2='GeV', show=show_plots, name='E_f at detector'
# )

# # t3.task('energy plot2')
# plib.plot_energy_std(
#     energies_i/1000, binsize=100, xlabel2='GeV', show=show_plots, name='E_i at Detector'
# )
#%%
plib.plot_hist(
    data_energy, 
    ylabel = '# of muons',
    xlabel1 = 'E',
    xlabel2 = 'GeV',
    label=r'$E_i \;(h=0)$',
    xlog=True,
    binsize=70,
    show_or_multiplot='multi',
    histtype='step'
)
bins_energies_i = plib.plot_hist(
    df['energies_i']/1000, 
    ylabel = '# of muons',
    xlabel1 = 'E',
    xlabel2 = 'GeV',
    label=r'$E_i \;(h=h_{\mathrm{det}})$',
    xlog=True,
    binsize=70,
    show_or_multiplot='multi',
    histtype='step'
)
plib.plot_hist(
    df['energies_f']/1000, 
    name=f'{config_file.file_name}_3hist',
    ylabel = '# of muons',
    xlabel1 = 'E',
    xlabel2 = 'GeV',
    label=r'$E_f \;(h=h_{\mathrm{det}})$',
    show_or_multiplot='show',
    xlog=True,
    binsize=70,
    savefig=True
    histtype='step'
)
#%%
plib.plot_hist(
    df['point2z']/100*(-1),
    ylabel = '# of muons',
    xlabel1 = 'd',
    xlabel2 = 'm',
    label=r'$muons$',
    name='z-coordinate of muons at detector',
    xlog=True,
    binsize=10,
    histtype='bar'
)
#%
plib.plot_hist(
    df['distances']/100/1000, 
    name='total distance of muons at detector',
    ylabel = '# of muons',
    xlabel1 = 'd',
    xlabel2 = 'km',
    label=r'$muons$',
    binsize=80,
    histtype='bar'
)


t3.stop(silent)



#%%

# theta full und 30° abschneiden hist plots 
############################################################
############################################################
############################################################
############################################################

file_name = 'EcoMug_std_full_1e4_xmom1e6.hdf'
(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        config_file.hdf_folder+file_name, f'main')
plib.plot_hist(
    np.degrees(data_theta), 
    ylabel = '# of muons',
    xlabel1 = '\theta',
    xlabel2 = '°',
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
        config_file.hdf_folder+file_name, f'main')
plib.plot_hist(
    np.degrees(data_theta), 
    ylabel='# of muons',
    xlabel1=r'\theta',
    xlabel2='°',
    name='',
    label=r'$\theta < 30°$',
    xlog=False,
    binsize=20,
    show_or_multiplot=True,
    savefig=True,
    histtype='step'
)
#%%
# ecomug daten roh plotten 
############################################################
############################################################
############################################################
############################################################
reload(plib)
file_name = "EcoMug_gaisser_30deg_1e7_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e4_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e5_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e6_min5e2_max3e5.hdf"

file_name = "EcoMug_gaisser_30deg_1e7_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e4_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e5_min5e2_max3e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e6_min2e2_max2e5.hdf"
file_name = "EcoMug_gaisser_30deg_1e7_min2e2_max2e5.hdf"
print(f'{file_name}')
(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        config_file.hdf_folder+file_name, f'main')
STATISTICS = len(data_energy)


t1 = stopwatch.stopwatch(title='full simulation: proposal init and simulation', time_unit='s')
t1.task('plot EcoMug data')
# plib.plot_energy_std(
#     data_energy, binsize=50,
#     xlabel2='GeV', show=show_plots)
plib.plot_hist(
    data_energy, 
    ylabel='# of muons',
    xlabel1=r'E',
    xlabel2='GeV',
    name='Ecomug_spektrum779',
    label='',
    xlog=True,
    binsize=40,
    show_and_nomultiplot=True,
    savefig=True
)
# Ecomug_spektrum779
t1.task('plot EcoMug data')

# plib.plot_hist(
#     np.degrees(data_phi), 
#     ylabel='# of muons',
#     xlabel1=r'\theta',
#     xlabel2='°',
#     name='',
#     label=r'$\theta < 30°$',
#     xlog=False,
#     binsize=20,
#     show_and_nomultiplot=True,
#     savefig=True,
#     histtype='step'
# )
# data = np.cos(data_theta)
# data = np.degrees(data_theta)
# plib.plot_hist(
#     data, 
#     ylabel='# of muons',
#     xlabel1=r'\theta',
#     xlabel2='°',
#     name=f'theta-{file_name}',
#     label=r'$\theta < 30°$',
#     xlog=False,
#     binsize=50,
#     show_and_nomultiplot=True,
#     savefig=True,
#     histtype='step'
# )
