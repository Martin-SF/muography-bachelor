# %%
# 1. import everything, load up EcoMug data
# from asyncio import exceptions
import os, sys
os.chdir(os.path.dirname(__file__))  # wichtig wenn nicht über ipython ausgeführt
#%%

from distributed import Client, LocalCluster, as_completed
# client = Client("localhost:8786")
client = Client("tcp://129.217.166.201:8786")
# client = Client("localhost:43887")
# client = Client()

import traceback
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from tqdm.gui import tqdm as tqdm_gui
import pandas as pd
# from numba import jit, njit, vectorize, prange
from importlib import reload
# import my_py_lib.my_plots_library as plib
# from my_py_lib.my_plots_library import pp_get_pos
def pp_get_pos(pos_obj):
    return [pos_obj.x, pos_obj.y, pos_obj.z]
from my_py_lib.stopwatch import stopwatch
# from stopwatch import stopwatch
import proposal as pp
print(os.getcwd())
# reload(stopwatch)
t = stopwatch()
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



#%%
# start
######################################################################
######################################################################
######################################################################
show_plots = True
# show_plots = False
STATISTICS = int(1e5)
STATISTICS = int(3e3)
STATISTICS = int(1e0)
print_results = False
t.settings(title='full proposal propagation and plotting')

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

t.task('read EcoMug data')
file_name = "EcoMug_fullspectrum.hdf"
file_name = "EcoMug_highenergy.hdf"
file_name = "EcoMug_highenergy_pos0.hdf"
file_name = "hdf_files/"+file_name
size = int(1e7)
data = pd.read_hdf(file_name, key=f'muons_{size}')[:STATISTICS]
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
# proposal-propagation
######################################################################
######################################################################
######################################################################
# reload(stopwatch)
# reload(plib)
t.task('initilaize propagation')
# STATISTICS = int(1e4)
distances = np.zeros(STATISTICS, dtype=float_type)
energies = np.zeros(STATISTICS, dtype=float_type)
energies2 = np.zeros(STATISTICS, dtype=float_type)
start_end_points = np.zeros(shape=(STATISTICS*2, 3), dtype=float_type)
muons = []

t.task('geometry define')
sizes1 = 20e2
detector_size = (sizes1, sizes1, sizes1)

t.task('propagation-loop')
counter = 0

print(f'before detector')
# max_distance, min_energy, hierarchy_condition
# max_distance *= 1e2*1e3 # cm in m in km
# max_distance = 1e2*10000  # cm
# print(os.getcwd())

pp.InterpolationSettings.tables_path = "/tmp"

# import configs.propagator as proper
import propagator as proper
# from propagator import prop_plus, prop_minus
t.task('propagation')
# def pp_propagate(position, energy_init, theta, phi, charge, **kwargs):
def pp_propagate(position, energy_init, theta, phi, charge):
    # print(os.getcwd())
    # position = (0,0,0)
    # t1 = stopwatch(
    # title='inside propagation loop', time_unit='µs',
    # selfexecutiontime_micros=0.7)  # total time when hit target 38 µs

   
    # os.chdir(kwargs['path'])  # notwendig!!

    propagate_settings = (1e20, 0, 10)  
    detector_pos = (0, 0, -1000e2)
    detector = pp.geometry.Cylinder(
        pp.Cartesian3D(detector_pos),
        inner_radius = 0,
        radius = 567200e2,
        height = 1e2
    )

    # t1.task()
    init_state = pp.particle.ParticleState()
    init_state.type = 13  # type for muons+

    # t1.task('give muon to proposal')  # 17% of loop time
    # init_state.momentum = momentum  # MeV
    init_state.energy = energy_init   # MeV
    init_state.position = pp.Cartesian3D(position)
    init_state.direction = pp.Cartesian3D(pp.Spherical3D(1, phi, theta))
    # t1.task('create propagator') # 20% of loop time


    # try:
    #     if (charge == 1):
    #         prop = pp.Propagator(
    #             particle_def=pp.particle.MuPlusDef(),
    #             path_to_config_file=config
    #             )
    #     else:
    #         prop = pp.Propagator(
    #             particle_def=pp.particle.MuMinusDef(),
    #             path_to_config_file=config
    #             )
    #     t1.task('propagator') # 20% of loop time
    #     track = prop.propagate(
    #             init_state, *propagate_settings)
    # except Exception as e:
    #     print(e)

    # proper = prop.prop_plus
    # track = proper.propagate(
    #             init_state, *propagate_settings)



    try:
        if (charge == 1):
            track = proper.prop_plus.propagate(
                init_state, *propagate_settings)
        else:
            track = proper.prop_minus.propagate(
                init_state, *propagate_settings)
    except Exception as e:
        print(e)

    

    # t1.task('did geometry hit?')  # 4% of loop time

    if (track.hit_geometry(detector) or True):
        hit_detector = True
        # t1.task('write propagate array and write to array')  # 14% loop time
        distance_at_track_end = track.track_propagated_distances()[-1]
        energy_at_track_end = track.track_energies()[-1]

        # t1.task('add start points to array')  # 41% of loop time TODO
        # point1 = plib.pp_get_pos(init_state.position)
        # point2 = plib.pp_get_pos(track.track()[-1].position)
        point1 = pp_get_pos(init_state.position)
        point2 = pp_get_pos(track.track()[-1].position)

    else:
        hit_detector = False
        distance_at_track_end = 0
        energy_at_track_end = 0
        point1 = 0
        point2 = 0

    # print(position, energy_init, theta, phi, charge)
    # hit_detector = False  
    # t1.stop()

    return (hit_detector, distance_at_track_end, energy_at_track_end, 
    energy_init, point1, point2)




def main():
    # print(f'outside{detector}')
    input = (data_position*10,
            data_energy*1e3,
            data_theta,
            data_phi,
            data_charge
            )
    # input = (
    #         data_energy*1e3,
    #         data_theta,
    #         data_phi,
    #         data_charge
    #         )
    # input = (np.array([0,0,0]),
    #         10000,
    #         0,
    #         0,
    #         1
    # )
        
    # client = Client()
    print(os.getcwd())
    # client.upload_file('my_py_lib/stopwatch.py')
    # client.upload_file('stopwatch.py')
    # client.upload_file('dask_lib.zip')
    # prop.config = "config_cylinder-huge.json"
    # client.upload_file(proper.config)
    client.upload_file('propagator.py')
    # client.upload_file('configs/propagator.py')
    # client.upload_file('propagator.py')
    # os.chdir(os.path.dirname(__file__))
    # futures = client.map(pp_propagate, *input,
    #             path = os.path.abspath(os.path.dirname(__file__)))
    batch = 6*6
    # futures = client.map(pp_propagate, *input, batch_size = int(batch), pure = False)
    futures = client.map(pp_propagate, *input, pure = True)
    results = client.gather(futures)

    # future = client.submit(pp_propagate, (0,0,0),
    #     10000,
    #     0,
    #     0,
    #     1,
    #     path = os.path.abspath(os.path.dirname(__file__)))
    # future = client.submit(pp_propagate, *input,
    #     path = os.path.abspath(os.path.dirname(__file__)))
    # results = future.result()

    # np_results = np.array(results)
    if (print_results):    
        print('print results')
        print(results)
    t.stop()


if __name__ == '__main__':
    main()
# %%

# %%
