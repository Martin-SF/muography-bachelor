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
t = stopwatch.stopwatch()
show_plots=True
import proposal as pp
float_type = np.float64

# import os, sys
# os.chdir(os.path.dirname(sys.argv[0]))
show_plots=False

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

t.settings(title = 'full proposal propagation and plotting')
t.task('read EcoMug data', True)
if 'data' not in locals():
    data = pd.read_hdf("EcoMug_fullspectrum.hdf", key=f'muons_{int(1e6)}')

t.task('transform_positioning position data', True)
# data_position = np.array(data['position'], dtype=float_type)  # geht nicht
# data_position = data['position']

def transform_position(arr):
    len_arr = len(arr)
    a = np.zeros(shape = (len_arr, 3), dtype=float_type)
    for i in range(len_arr):
        a[i] = np.array(arr[i], dtype=float_type)
    return a

data_position = transform_position(data['position'])

# data_x = data['position']
# data_y = data['position']
# data_z = data['position']

data_momentum = np.array(data['momentum'], dtype=float_type) 
data_theta = np.array(data['theta'], dtype=float_type) 
data_phi = np.array(data['phi'], dtype=float_type)
data_charge = np.array(data['charge'], dtype= np.uint32)


t.task('plot data', True)
energy_readout = data['energy'] # [0:1000]*1000
plib.plot_energy_std(energy_readout, binsize = 50, xlabel_unit = 'GeV', show=show_plots)
# t.stop()
# %%
# load proposal propagators, building interpolation tables...
######################################################################
######################################################################
t.task('create prop_minus')

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

# t.settings(title = 'load propagators', unit = 's')
pp.InterpolationSettings.tables_path = "/tmp"
prop_minus = pp.Propagator(
	  particle_def=pp.particle.MuMinusDef(),
	  path_to_config_file=config
)
t.task('create prop_plus')
prop_plus = pp.Propagator(
	  particle_def=pp.particle.MuPlusDef(),
	  path_to_config_file=config
)
init_state = pp.particle.ParticleState()
init_state.type = 13  # type for muons+
# t.stop()
#%
# proposal-propagation
######################################################################
######################################################################
######################################################################
reload(stopwatch)
reload(plib)
t.task('initilaize propagation')
STATISTICS = int(1e6)
distances = np.zeros(STATISTICS, dtype=float_type)
energies = np.zeros(STATISTICS, dtype=float_type)
start_end_points = np.zeros(shape = (STATISTICS*2,3), dtype=float_type)
muons = []

max_distance = 1e20
# max_distance *= 1e2*1e3 # cm in m in km
min_energy = 0  # MeV
# max_distance = 1e2*10000  # cm
hierarchy_condition = 10

t.task('geometry define')  # 6% of loop time
# detector = pp.geometry.Box(pp.Cartesian3D(0, 0, -500e2), 100e2, 100e2, 100e2)
# detector = pp.geometry.Box(pp.Cartesian3D(0, 0, -100), 100, 100, 100)
detector_pos = (0, 0, -100e2)
sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)
# detector = pp.geometry.Box(pp.Cartesian3D(*detector_pos), *detector_size)
detector = pp.geometry.Cylinder(pp.Cartesian3D(*detector_pos), 0, 150e2, 10e2)

t.task('propagation-loop', True)
counter = 0
t1 = stopwatch.stopwatch(title='inside propagation loop', time_unit='µs'
                        , selfexecutiontime_in_ms = 0.9)
a = np.empty(3, dtype=float_type)
plib.pp_get_pos(init_state.position)
start_end_points[0] = 1

for event in tqdm(range(STATISTICS), disable=False):
    # idee: data in ein 5D array packen wo momentum in [0] position in [1] usw...?
    # data arrays sind in float_type 64. vielleicht besser ungenaueren float_type zu benutzen
    # momentum = data['momentum'][event]*1e3
    # position = data['position'][event]
    # theta = data['theta'][event]
    # phi = data['phi'][event]
    # charge = data['charge'][event]
    t1.task('read data pos')  # 2% of loop time TODO
    position = data_position[event]
    t1.task('read mom 1')  # 2% of loop time TODO
    momentum = data_momentum[event]*1e3
    t1.task('read momentum 2')  # 2% of loop time TODO
    momentum = data['momentum'][event]*1e3

    t1.task('read theta')  # 2% of loop time TODO
    theta = data_theta[event]
    t1.task('read phi')  # 2% of loop time TODO
    phi = data_phi[event]
    t1.task('read chargw')  # 2% of loop time TODO
    charge = data_charge[event]
    t1.task('rest')  # 2% of loop time TODO
    # current_muon = f'[{event}]: position: {position} direction: (1, {phi}, {theta}) momentum (MeV): {momentum} charge: {charge}\n'

    # t1.task('give muon to proposal')  # 19% of loop time
    init_state.momentum = momentum  # initial momentum in MeV
    init_state.position = pp.Cartesian3D(position)
    init_state.direction = pp.Cartesian3D(pp.Spherical3D(1, phi, theta))
    # optimize: get initial position from dataframe not from proposal
    # t1.task('propagation') # 19% of loop time
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
    
    # t1.task('did geometry hit?')  # 3% of loop time
    if (track.hit_geometry(detector) or False):
        
        # t1.task('write propagate array and write to array')  # 14% of loop time
        distance_at_track_end = track.track_propagated_distances()[-1]
        energy_at_track_end = track.track_energies()[-1]
        if (distance_at_track_end < 100e2):
            # t1.stop(silent = True)
            continue
        distances[counter] = distance_at_track_end
        energies[counter] = energy_at_track_end 

        # t1.task('add to array start point 1')  # 23% of loop time TODO
        start_end_points[counter*2] = plib.pp_get_pos(init_state.position)
        # t1.task('add to array start point 2')  # 18% of loop time TODO
        start_end_points[counter*2+1] = plib.pp_get_pos(track.track()[-1].position)

        # t1.task()
        # t1.stop(silent = True)
        t1.stop()
        counter += 1
        break
    t1.stop(silent = True)
# quit()
# t.stop()
# %%
t.task('deleting arrays')
print(f'{counter} of {STATISTICS} muons ({counter/STATISTICS*100:.4}%) did hit the defined geometry')
start_end_points = np.delete(start_end_points, np.s_[(counter*2):], 0)
distances = np.delete(distances, np.s_[(counter):], 0)
energies = np.delete(energies, np.s_[(counter):], 0)

t.task('writing muons.txt')
# test = np.array([[0,0,0],[1,1,1],[0,0,0],[2,2,2],[0,0,0],[3,3,3],[0,0,0],[4,4,4],[0,0,0],[5,5,5],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[99,99,99]])
with open('muons.txt', 'w') as f:
    f.writelines(muons) 

# t.stop()
# %
# plots
######################################################################
######################################################################
######################################################################
######################################################################
t.task('reload plib')
reload(plib)
t.settings(title = None)
t.task('3D plot')
plib.plot_3D_start_end(start_end_points[:int(1e5)], detector_pos, detector_size, 
                        elev=10, azim=70, alpha=0.3, dpi=1, show=show_plots, title = f'# of particles: {counter}')

t.task('distances plot')
plib.plot_distances_std(distances/100, 100, xlabel_unit='m', show=show_plots)

t.task('energy plot')
plib.plot_energy_std(energies/1000, binsize = 100, xlabel_unit='GeV', show=show_plots)
t.stop()

# 
# %%

# %%
t.task()
t.task()
t.stop(silent = True)
t.task()
t.task()
t.stop()
t.task()
t.task()
t.stop()


# %%
