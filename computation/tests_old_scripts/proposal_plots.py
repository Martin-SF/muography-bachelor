# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import os, sys
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# os.chdir(os.path.dirname(sys.argv[0]))

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

import proposal as pp
pp.InterpolationSettings.tables_path = "/tmp"  # save interpolation tables to folder
config = "config_minimal.json"

prop = pp.Propagator(
	  particle_def=pp.particle.MuMinusDef(),
	  path_to_config_file=config
)
init_state = pp.particle.ParticleState()
init_state.type = 13  # type for muons+

from my_plots_library import *

# %%
STATISTICS = int(1e3)
distances = np.empty(STATISTICS, dtype=float)
energies = np.empty(STATISTICS, dtype=float)
start_end_points = np.empty(shape = (STATISTICS*2,3), dtype=float)
fixed_energy = 0  # MeV
fixed_distance = 0.01
fixed_distance *= 1e2*1e3  # cm in m in km

position = pp.Cartesian3D(0, 0, 0)
phi = 0
momentum = 1e9  # 1e3 GeV; 1e6 1 TeV; 1 PeV
charge = -1

init_state.position = pp.Cartesian3D(position)
init_state.momentum = momentum  # initial momentum in MeV


for event in tqdm(range(STATISTICS)):
    theta = 0
    theta = np.random.rand(1)
    theta += np.pi

    init_state.direction = pp.Cartesian3D(pp.Spherical3D(1, phi, theta))

    start_end_points[event*2] = pp_get_pos(init_state.position)
    track = prop.propagate(init_state, fixed_distance, fixed_energy)
    start_end_points[event*2+1] = pp_get_pos(track.track()[-1].position)

    distance_at_track_end = track.track_propagated_distances()[-1]
    energy_at_track_end = track.track_energies()[-1]
    distances[event] = distance_at_track_end
    energies[event] = energy_at_track_end
    

distances /= 100  # cm in m

plot_3D_start_end(start_end_points, azim=20, alpha=0.1)
# plot_distances_std(distances, 100)
# plot_energy_std(energies, 1000)

# track.track()[-1].position.z


