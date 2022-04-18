# %%
# 1. import everything, load up EcoMug data
# from asyncio import exceptions
import traceback
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os, sys
os.chdir(os.path.dirname(__file__))
# from numba import jit, njit, vectorize, prange
from importlib import reload
import my_py_lib.my_plots_library as plib
import my_py_lib.stopwatch as stopwatch
import proposal as pp
reload(stopwatch)
t = stopwatch.stopwatch()
show_plots = True
float_type = np.float64


def transform_position_list(arr):
    len_arr = len(arr)
    a = np.zeros(shape=(len_arr, 3), dtype=float_type)
    for i in range(len_arr):
        a[i] = np.array(arr[i], dtype=float_type)
    return a


def transform_position_posxyz(x, y, z):
    a = np.zeros(shape=(len(x), 3), dtype=float)
    a[:, 0] = x
    a[:, 1] = y
    a[:, 2] = z
    return a



# show_plots = False

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

t.settings(title='full proposal propagation and plotting')
t.task('read EcoMug data')
file_name = "EcoMug_fullspectrum.hdf"
file_name = "EcoMug_highenergy.hdf"
file_name = "EcoMug_highenergy_pos0.hdf"
file_name = "hdf_files/"+file_name
size = int(1e7)
data = pd.read_hdf(file_name, key=f'muons_{size}')
# if 'data' not in locals():
#     data = pd.read_hdf(file_name, key=f'muons_{size}')

t.task('transform_positioning position data')
# data_position = transform_position_list(data['position'])
data_position = transform_position_posxyz(
    data['pos_x'], data['pos_y'], data['pos_z']
    )

data_momentum = np.array(data['momentum'], dtype=float_type)
data_energy = np.array(data['energy'], dtype=float_type)
data_theta = np.array(data['theta'], dtype=float_type)
data_phi = np.array(data['phi'], dtype=float_type)
data_charge = np.array(data['charge'], dtype=np.uint32)


t.task('plot data')
energy_readout = data['energy']
# plib.plot_energy_std(
#     energy_readout, binsize=50,
#     xlabel_unit='GeV', show=show_plots)
# t.stop()
# %
# load proposal propagators, building interpolation tables...
######################################################################
######################################################################
t.task('create prop_minus and plus')

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
    # config = "config_min_muo2.json"
}
config = "config_kirchhellen_sandstein.json"
config = "configs/"+config


pp.InterpolationSettings.tables_path = "/tmp"
prop_minus = pp.Propagator(
    particle_def=pp.particle.MuMinusDef(),
    path_to_config_file=config
)
prop_plus = pp.Propagator(
    particle_def=pp.particle.MuPlusDef(),
    path_to_config_file=config
)
init_state = pp.particle.ParticleState()
init_state.type = 13  # type for muons+
# t.stop()
# %#
# proposal-propagation
######################################################################
######################################################################
######################################################################
reload(stopwatch)
reload(plib)
t.task('initilaize propagation')
STATISTICS = int(1e5)
distances = np.zeros(STATISTICS, dtype=float_type)
energies = np.zeros(STATISTICS, dtype=float_type)
energies2 = np.zeros(STATISTICS, dtype=float_type)
start_end_points = np.zeros(shape=(STATISTICS*2, 3), dtype=float_type)
muons = []

# max_distance, min_energy, hierarchy_condition
propagate_settings = (1e20, 0, 10)  

t.task('geometry define')
sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)
# detector = pp.geometry.Box(pp.Cartesian3D(*detector_pos), *detector_size)

# detector_pos = (0, 0, -99e2)
# detector = pp.geometry.Cylinder(
#     pp.Cartesian3D(detector_pos),
#     inner_radius = 0,
#     radius = 50e2,
#     height = 2e2
# )
detector_pos = (0, 0, -1000e2)
detector = pp.geometry.Cylinder(
    pp.Cartesian3D(detector_pos),
    inner_radius = 0,
    radius = 1e20,
    height = 1e2
)

t.task('propagation-loop', True)
counter = 0
# t1 = stopwatch.stopwatch(
#     title='inside propagation loop', time_unit='µs',
#     selfexecutiontime_micros=0.7)  # total time when hit target 38 µs

for event in tqdm(range(STATISTICS), disable=False):
    # t1.task('read data')  # 3% of loop time TODO
    position = data_position[event]*10
    # momentum = data_momentum[event]*1e3
    energy = data_energy[event]*1e3*1e3
    theta = data_theta[event]
    phi = data_phi[event]
    charge = data_charge[event]

    # t1.task('give muon to proposal')  # 17% of loop time
    # init_state.momentum = momentum  # MeV
    init_state.energy = energy   # MeV
    init_state.position = pp.Cartesian3D(position)
    init_state.direction = pp.Cartesian3D(pp.Spherical3D(1, phi, theta))
    # t1.task('propagation') # 20% of loop time
    try:
        if (charge == 1):
            track = prop_plus.propagate(
                init_state, *propagate_settings)
        else:
            track = prop_minus.propagate(
                init_state, *propagate_settings)
    except Exception as e:
        print(e)
        current_muon = (f'[{event}], {position}, (1, {phi}, {theta}), ' +
            f'{energy} MeV, {charge}\n')
        print(f'failed muon: {current_muon}')
        traceback.print_exc()
        muons.append(current_muon)
        break

    # t1.task('did geometry hit?')  # 4% of loop time
    if (track.hit_geometry(detector) or False):
        # t1.task('write propagate array and write to array')  # 14% loop time
        distance_at_track_end = track.track_propagated_distances()[-1]
        energy_at_track_end = track.track_energies()[-1]
        # if (distance_at_track_end < 100e2):
        #     # t1.stop(silent = True)
        #     continue
        distances[counter] = distance_at_track_end
        energies[counter] = energy_at_track_end 
        energies2[counter] = energy

        # t1.task('add start points to array')  # 41% of loop time TODO
        start_end_points[counter*2] = plib.pp_get_pos(init_state.position)
        start_end_points[counter*2+1] = plib.pp_get_pos(
                                            track.track()[-1].position)

        # t1.stop(silent = True)
        # t1.stop()
        counter += 1
        # break
    # t1.stop(silent=True)

# t.stop()
# %
t.task('deleting arrays')
print(
    f'{counter} of {STATISTICS} muons ({counter/STATISTICS*100:.4}%) ' +
    'hit the defined geometry'
)
start_end_points = np.delete(start_end_points, np.s_[(counter*2):], 0)
distances = np.delete(distances, np.s_[(counter):], 0)
energies = np.delete(energies, np.s_[(counter):], 0)
energies2 = np.delete(energies2, np.s_[(counter):], 0)

test = np.array(
    [
        [0, 0, 0], [1, 1, 1], [0, 0, 0],
        [2,  2, 2], [0, 0, 0], [3, 3, 3],
        [0, 0, 0], [4, 4, 4], [0, 0, 0],
        [5, 5, 5], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [99, 99, 99]
    ]
)
t.task('writing muons.txt')
with open('muons.txt', 'w') as f:
    f.writelines(muons)

t.stop()
# %%
# plots
######################################################################
######################################################################
######################################################################
######################################################################
reload(plib)
t.settings(title=None)
t.task('3D plot')
plib.plot_3D_start_end(
    start_end_points, detector_pos, detector_size,
    elev=10, azim=70, alpha=0.3, dpi=1, show=show_plots,
    title=f'# of particles: {counter}'
)
# t.task('distances plot')
# plib.plot_distances_std(
#     distances/100, 100, xlabel_unit='m', show=show_plots
# )
# t.task('energy plot')
# plib.plot_energy_std(
#     energies/1000, binsize=100, xlabel_unit='GeV', show=show_plots
# )

# t.task('energy plot2')
# plib.plot_energy_std(
#     energies2/1000, binsize=100, xlabel_unit='GeV', show=show_plots
# )
t.stop(silent=True)
