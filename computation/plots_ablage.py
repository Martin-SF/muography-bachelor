import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
# from scipy.stats import norm
# import proposal as pp
from EcoMug_pybind11.build import EcoMug
# from numba import vectorize
# from numba import jit, njit, prange
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
from importlib import reload
reload(slib)
reload(plib)
reload(stopwatch)




# theta full und 30° abschneiden hist plots 
############################################################
############################################################
############################################################
############################################################

file_name = 'EcoMug_std_full_1e4_xmom1e6.hdf'
(data_position, data_momentum, data_energy,
    data_theta, data_phi, data_charge) = slib.read_muon_data(
        "data_hdf/"+file_name, f'main')
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
        "data_hdf/"+file_name, f'main')
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
