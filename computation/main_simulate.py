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
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
import proposal as pp

reload(stopwatch)
t = stopwatch.stopwatch()
show_plots = True
# show_plots = False
FLOAT_TYPE = np.float64

t.settings(title='full proposal propagation and plotting')
t.task('read EcoMug data')
size = '1e7'
param = 'std'
param = 'guan'
param = 'gaisser'
angle = '30deg'
angle = 'full'
file_name = f'EcoMug_{param}_{angle}_{size}.hdf'
size = int(float(size))
(data_position, data_momentum, data_energy,
    data_theta, data_phi, 
    data_charge) = slib.read_muon_data(
        "data_hdf/"+file_name, f'main')

t.task('plot data')
# plib.plot_energy_std(
#     data_energy, binsize=50,
#     xlabel_unit='GeV', show=show_plots)
# plib.plot_distances_std(
#     data_theta, binsize=50, name='theta dist ',
#     xlabel_unit='theta', show=show_plots)
# plib.plot_distances_std(
#     slib.change_zenith_convention(data_theta), binsize=50, name='theta dist changed ',
#     xlabel_unit='theta', show=show_plots)
# plib.plot_distances_std(
#     np.cos(data_theta), binsize=50, name='theta dist cos',
#     xlabel_unit='cos(theta)', show=show_plots)
# plib.plot_distances_std(
#     np.cos(slib.change_zenith_convention(data_theta)), binsize=50, name='theta dist changed cos',
#     xlabel_unit='cos(theta)', show=show_plots)
# %%
# load proposal propagators, building interpolation tables...
######################################################################
######################################################################
t.task('create prop_minus and plus')
{
    # config = "config_earth.json"
    # config = "config_full_onesector_nomultiplescattering.json"
    # config = "config_full_onesector.json"
    # config = "config_full.json"
    # config = "config_minimal.json"
    # config = "config_muo0_r.json"
    # config = "config_muo0.json"
    # config = "config_min_muo.json"
    # config = "config_min_muo2.json"
    # config = "config_cylinder-huge.json"
}
config = "sandstein.json"
config = "stdrock.json"

pp.InterpolationSettings.tables_path = "/tmp"
prop_minus = pp.Propagator(
    particle_def=pp.particle.MuMinusDef(),
    path_to_config_file="config/"+config
)
prop_plus = pp.Propagator(
    particle_def=pp.particle.MuPlusDef(),
    path_to_config_file="config/"+config
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
STATISTICS = int(1e3)
distances = np.zeros(STATISTICS, dtype=FLOAT_TYPE)
energies = np.zeros(STATISTICS, dtype=FLOAT_TYPE)
energies2 = np.zeros(STATISTICS, dtype=FLOAT_TYPE)
start_end_points = np.zeros(shape=(STATISTICS*2, 3), dtype=FLOAT_TYPE)
muons = []

# max_distance, min_energy, hierarchy_condition
propagate_settings = (1e20, 0, 10)  

t.task('define detector')
sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)
# detector = pp.geometry.Box(pp.Cartesian3D(*detector_pos), *detector_size)

# detector = pp.geometry.Cylinder(
#     pp.Cartesian3D(detector_pos), inner_radius = 0,
#     radius = 50e2, height = 2e2 )

# detector_pos = (0, 0, -99e2)
# detector_pos = (0, 0, -1205e2)
detector_pos = (0, 0, -500e2)
detector = pp.geometry.Cylinder(
    pp.Cartesian3D(detector_pos), inner_radius = 0,
    radius = 1e20, height = 2e2
)

t.task('propagation-loop', True)
counter = 0
# t1 = stopwatch.stopwatch(
#     title='propagation loop', time_unit='µs',
#     selfexecutiontime_micros=0.7)  # total time when hit target 38 µs

position = np.array([0,0,0])
init_state.position = pp.Cartesian3D(position)
# data_theta = slib.change_zenith_convention(data_theta)  # dann fliegen sie nach oben
for event in tqdm(range(STATISTICS), disable=False):
    # t1.task('read data')  # 3% of loop time TODO
    energy = data_energy[event]*1e3
    theta = data_theta[event]
    phi = data_phi[event]
    charge = data_charge[event]

    # t1.task('give muon to proposal')  # 17% of loop time
    init_state.energy = energy   # MeV
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
        traceback.print_exc()
        current_muon = (f'[{event}], {position}, (1, {phi}, {theta}), ' +
            f'{energy} MeV, {charge}\n')
        print(f'failed muon: {current_muon}')
        muons.append(current_muon)
        # break

    # t1.task('did geometry hit?')  # 4% of loop time
    if (track.hit_geometry(detector) or True):
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
start_end_points = np.delete(start_end_points, np.s_[(counter*2):], 0)
distances = np.delete(distances, np.s_[(counter):], 0)
energies = np.delete(energies, np.s_[(counter):], 0)
energies2 = np.delete(energies2, np.s_[(counter):], 0)

print(
    f'{counter} of {STATISTICS} muons ({counter/STATISTICS*100:.4}%) ' +
    'hit the detector'
    )
print(f'min E_i at detector is {min(energies2)/1000} GeV')
t.task('writing muons.txt')
with open('muons.txt', 'w') as f:
    f.writelines(muons)
# t.stop()
# %
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
plib.plot_energy_std(
    energies/1000, binsize=100, xlabel_unit='GeV', show=show_plots, name='E_f at detector'
)

# t.task('energy plot2')
# plib.plot_energy_std(
#     energies2/1000, binsize=100, xlabel_unit='GeV', show=show_plots, name='E_i at Detector'
# )
t.stop(silent=True)

# %%
