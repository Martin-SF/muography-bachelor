# %%
# 1. import everything, load up EcoMug data
from asyncio import exceptions
import traceback
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from tqdm.gui import tqdm as tqdm_gui
import pandas as pd
from numba import jit, njit, vectorize, prange
from importlib import reload
import my_py_lib.my_plots_library as plib
import my_py_lib.stopwatch as stopwatch
reload(stopwatch)
t = stopwatch.stopwatch(print_all_tasks=True)
show_plots=True
show_plots=False
import proposal as pp

# import os, sys
# os.chdir(os.path.dirname(sys.argv[0]))

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

# t.start(title = 'read EcoMug data')
t.start(title = 'full proposal propagation and plotting')
t.task('load data')
if 'data' not in locals():
    data = pd.read_hdf("EcoMug_fullspectrum.hdf", key=f'muons_{int(1e6)}')
t.task('plot data')
energy_readout = data['energy']  # [0:1000]*1000
plib.plot_energy_std(energy_readout, binsize = 50, xlabel_unit = 'GeV', show=show_plots)
# t.stop()
# %%
# load proposal propagators, building interpolation tables...
######################################################################
######################################################################

{
# config = "config_earth.json"
# config = "config_full_edit.json"
# config = "config_full_edit2.json"
# config = "config_full_onesector_nomultiplescattering.json"
# config = "config_full_onesector.json"
# config = "config_full.json"
# config = "config_minimal.json"
# config = "config_muo0_r.json"
# config = "config_muo0.json"
# config = "config_min_muo.json"
}
config = "config_min_muo2.json"

# t.start(name = 'prop_minus', title = 'load propagators', unit = 's')
t.start('prop_minus')
pp.InterpolationSettings.tables_path = "/tmp"
prop_minus = pp.Propagator(
	  particle_def=pp.particle.MuMinusDef(),
	  path_to_config_file=config
)
t.task('prop_plus')
# t.stop()
# prop_plus = prop_minus
prop_plus = pp.Propagator(
	  particle_def=pp.particle.MuPlusDef(),
	  path_to_config_file=config
)
init_state = pp.particle.ParticleState()
init_state.type = 13  # type for muons+
t.stop()
#%%
# proposal-propagation
######################################################################
######################################################################
######################################################################
t.start('initilaize loop')
STATISTICS = int(1e6)
distances = np.zeros(STATISTICS, dtype=float)
energies = np.zeros(STATISTICS, dtype=float)
start_end_points = np.zeros(shape = (STATISTICS*2,3), dtype=float)
muons = []

max_distance = 1e20
# max_distance *= 1e2*1e3 # cm in m in km
min_energy = 0  # MeV
# max_distance = 1e2*10000  # cm
hierarchy_condition = 100
t.task('propagation-loop')
counter = 0
for event in tqdm(range(STATISTICS), disable=False):
    position = data['position'][event]
    momentum = data['momentum'][event]*1e3
    theta = data['theta'][event]
    phi = data['phi'][event]
    charge = data['charge'][event]
    # current_muon = f'[{event}]: position: {position} direction: (1, {phi}, {theta}) momentum (MeV): {momentum} charge: {charge}\n'

    init_state.position = pp.Cartesian3D(position)
    init_state.momentum = momentum  # initial momentum in MeV
    # optimize: maybe faster calculating energy on my end, than pp?
    init_state.direction = pp.Cartesian3D(pp.Spherical3D(1, phi, theta))
    # optimize: get initial position from dataframe not from proposal
    try:
        if (charge == 1):
            track = prop_plus.propagate(init_state, max_distance, min_energy, hierarchy_condition)
        else:
            track = prop_minus.propagate(init_state, max_distance, min_energy, hierarchy_condition)
    except Exception as e:
        print(e)
        current_muon = f'[{event}], {position}, (1, {phi}, {theta}), {momentum} MeV, {charge}\n'
        print(f'failed muon: {current_muon}')
        traceback.print_exc()
        muons.append(current_muon)
        break
    
    # detector = pp.geometry.Box(pp.Cartesian3D(0, 0, -500e2), 100e2, 100e2, 100e2)
    # detector = pp.geometry.Box(pp.Cartesian3D(0, 0, -100), 100, 100, 100)
    detector_pos = (0, 0, -100e2)
    sizes1 = 20e2
    detector_size = (sizes1, sizes1, sizes1)
    # detector = pp.geometry.Box(pp.Cartesian3D(*detector_pos), *detector_size)
    detector = pp.geometry.Cylinder(pp.Cartesian3D(*detector_pos), 0, 15e2, 1e2)
    if (track.hit_geometry(detector) or True):
        
        distance_at_track_end = track.track_propagated_distances()[-1]
        energy_at_track_end = track.track_energies()[-1]
        if (distance_at_track_end < 100e2):
            continue
        distances[counter] = distance_at_track_end
        energies[counter] = energy_at_track_end 
        start_end_points[counter*2] = plib.pp_get_pos(init_state.position)
        start_end_points[counter*2+1] = plib.pp_get_pos(track.track()[-1].position)
        counter += 1
t.task('deleting arrays')
print(f'{counter} of {STATISTICS} muons ({counter/STATISTICS*100:.4}%) did hit the defined geometry')
start_end_points = np.delete(start_end_points, np.s_[(counter*2):], 0)
distances = np.delete(distances, np.s_[(counter):], 0)
energies = np.delete(energies, np.s_[(counter):], 0)

t.task('writing muons.txt')
# test = np.array([[0,0,0],[1,1,1],[0,0,0],[2,2,2],[0,0,0],[3,3,3],[0,0,0],[4,4,4],[0,0,0],[5,5,5],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[99,99,99]])
with open('muons.txt', 'w') as f:
    f.writelines(muons) 

# %%
# plots
######################################################################
######################################################################
######################################################################
######################################################################
t.task('reload plib')
reload(plib)
t.task('3D plot')

plib.plot_3D_start_end(start_end_points[:int(1e5)], detector_pos, detector_size, 
                        elev=10, azim=70, alpha=0.3, dpi=1, show=show_plots, title = f'# of particles: {counter}')

# t.task('distances plot')
# plib.plot_distances_std(distances/100, 100, xlabel_unit='m', show=show_plots)

# t.task('energy plot')
# plib.plot_energy_std(energies/1000, binsize = 100, xlabel_unit='GeV', show=show_plots)
t.stop()

# 
# %%

# %%

# %%
