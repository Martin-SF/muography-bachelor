# %%
# import everything, building interpolation tables...
# from xml.dom.expatbuilder import theDOMImplementation
from asyncio import exceptions
import traceback
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import random
import os, sys
from numba import jit, njit, vectorize, prange
# import EcoMug

# os.chdir(os.path.dirname(sys.argv[0]))

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

import proposal as pp
pp.InterpolationSettings.tables_path = "/tmp"  # save interpolation tables to folder

# config = "config_muography0_r.json"  # config_minimal  config_muography0
# config = "config_minimal.json"
# config = "config_earth.json"
# config = "config_full_edit.json"
# config = "config_full_edit2.json"
# config = "config_full_onesector_nomultiplescattering.json"
config = "config_full_onesector.json"
config = "config_muography0.json"
config = "config_full.json"

prop_minus = pp.Propagator(
	  particle_def=pp.particle.MuMinusDef(),
	  path_to_config_file=config
)
# prop_plus = prop_minus
prop_plus = pp.Propagator(
	  particle_def=pp.particle.MuPlusDef(),
	  path_to_config_file=config
)
init_state = pp.particle.ParticleState()
init_state.type = 13  # type for muons+

# %%
# read EcoMug data
######################################################################
######################################################################
######################################################################
######################################################################

file_name2 = "EcoMug_fullspectrum.hdf"
STATISTICS = int(1e6)
data = pd.read_hdf(file_name2, key=f'muons_{STATISTICS}')
# %%
# plot ecomug data
energies = data['energy'][0:1000]*1000
bins = np.geomspace(min(energies), max(energies), 40)
plt.xscale('log')
plt.xlabel(r'$E \,/\, \mathrm{MeV} $')
plt.ylabel("Frequency")
_ = plt.hist(energies, bins=bins, log=True) 
# %%
# proposal-propagation
######################################################################
######################################################################
######################################################################
######################################################################

STATISTICS = int(1e5)

# distances = [float]*STATISTICS
# energies = [float]*STATISTICS
energies = []
muons = []*STATISTICS
min_energy = 0  # MeV
# fixed_energy = 100  # MeV
max_distance = 1e20
# max_distance = 1e2*10000  # cm
hierarchy_condition = 2

errors = 0
for event in tqdm(range(STATISTICS), disable=False):
    position = data['position'][event]
    momentum = data['momentum'][event]*1e3
    theta = data['theta'][event]
    phi = data['phi'][event]
    charge = data['charge'][event]
    # current_muon = f'[{event}]: position: {position} direction: (1, {phi}, {theta}) momentum (MeV): {momentum} charge: {charge}\n'
    current_muon = f'[{event}], {position}, (1, {phi}, {theta}), {momentum} MeV, {charge}\n'
    muons.append(current_muon)

    init_state.position = pp.Cartesian3D(position)
    init_state.momentum = momentum  # initial momentum in MeV
    init_state.direction = pp.Cartesian3D(pp.Spherical3D(1, phi, theta))
    try:
        if (charge == 1):
            track = prop_plus.propagate(init_state, max_distance, min_energy, hierarchy_condition)
        else:
            track = prop_minus.propagate(init_state, max_distance, min_energy, hierarchy_condition)
    except Exception as e:
        print(e)
        print(f'failed muon: {current_muon}')
        # traceback.print_exc()
    # track.hit_geometry(pp.geometry.Box(pp.Vector3D(0,0,0)))
    if (track.hit_geometry(pp.geometry.Box(pp.Cartesian3D(0, 0, 500e2), 100e3, 100e3, 100e3))):
        # distance_at_track_end = track.track_propagated_distances()[-1] / 100  # in m
        energy_at_track_end = track.track_energies()[-1]  # in MeV
        track.track()
        # distances.append(distance_at_track_end)
        energies.append(energy_at_track_end)
        # distances[event] = distance_at_track_end
        # energies[event] = energy_at_track_end


with open('muons.txt', 'w') as f:
    f.writelines(muons) 

bins = np.geomspace(min(energies), max(energies), 40)
plt.xscale('log')
plt.xlabel(r'$E \,/\, \mathrm{MeV} $')
plt.ylabel("Frequency")
_ = plt.hist(energies, bins=bins, log=True) 

# plt.show()
# plt.close()

# bins = np.linspace(min(distances), max(distances), 20)
# # plt.xscale('log')
# plt.xlabel(r'propagated distance $\,/\, \mathrm{m} $')
# plt.ylabel("Frequency")
# _ = plt.hist(distances, bins=bins, log=True)

# plt.show()
# %%
