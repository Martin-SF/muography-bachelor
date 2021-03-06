#%%
import traceback
import proposal as pp
import numpy as np
import py_library.my_plots_library as plib
import py_library.stopwatch as stopwatch
import py_library.simulate_lib as slib
import os, sys
import config as config_file
from importlib import reload

reload(config_file)
# os.chdir(os.path.dirname(__file__))

# print(f'PP_config : {config_file.PP_config}')

detector = pp.geometry.Cylinder(
    pp.Cartesian3D(config_file.detector_pos),
    inner_radius = 0,
    radius = 1e20,
    height = 1e2
)

pp.RandomGenerator.get().set_seed(int(np.random.random()*10000))
pp.InterpolationSettings.nodes_utility = 1000
pp.InterpolationSettings.tables_path = config_file.pp_tables_path

prop_plus = pp.Propagator(
    particle_def=pp.particle.MuPlusDef(),
    path_to_config_file=config_file.path_to_config_file
)

prop_minus = pp.Propagator(
    particle_def=pp.particle.MuMinusDef(),
    path_to_config_file=config_file.path_to_config_file
)

init_state = pp.particle.ParticleState()
init_state.type = 13  # type for muons+
position = np.array([0,0,0])
init_state.position = pp.Cartesian3D(position)
# def pp_propagate(energy_init, theta):
# def pp_propagate(energy_init, theta, phi, charge):
def pp_propagate(input):
    energy_init = input[0]
    theta = input[1]
    phi = input[2]
    charge = input[3]
    # current_muon = (f'{position}, (1, {phi}, {theta}), {energy_init*1000}MeV, {charge}\n')
    # print(f'{current_muon}')
    # print(os.getcwd())
    # t1 = stopwatch(
    # title='inside propagation loop', time_unit='µs',
    # selfexecutiontime_micros=0.7)  # total time when hit target 38 µs

    # t1.task('give muon to proposal')  # 17% of loop time
    # init_state.momentum = momentum  # MeV
    energy_init = energy_init*1000
    init_state.energy = energy_init   # MeV
    # init_state.position = pp.Cartesian3D(position)
    init_state.direction = pp.Cartesian3D(pp.Spherical3D(1, phi, theta))

    try:
        if (charge == 1):
            track = prop_plus.propagate(
                init_state, *config_file.propagate_settings)
        else:
            track = prop_minus.propagate(
                init_state, *config_file.propagate_settings)
    except Exception as e:
        print(e)
        traceback.print_exc()
        # current_muon = (f'[], {position}, (1, {phi}, {theta}), ' +
        #     f'{energy} MeV, {charge}\n')
        # print(f'failed muon: {current_muon}')
        # muons.append(current_muon)

    # t1.task('did geometry hit?')  # 4% of loop time
    # if (track.hit_geometry(detector) or False):
    point2 = plib.pp_get_pos(track.track()[-1].position)
    if (point2[2] <= config_file.detector_pos[2]):
        hit_detector = True
        # t1.task('write propagate array and write to array')  # 14% loop time
        distance_at_track_end = track.track_propagated_distances()[-1]
        energy_at_track_end = track.track_energies()[-1]

        # t1.task('add start points to array')  # 41% of loop time TODO
        point1 =  plib.pp_get_pos(init_state.position)
        # point1 = pp_get_pos(init_state.position)
        # point2 = pp_get_pos(track.track()[-1].position)
        point1x=point1[0]
        point1y=point1[1]
        point1z=point1[2]
        point2x=point2[0]
        point2y=point2[1]
        point2z=point2[2]
    else:
        # print(track.track()[-1].position.y)
        # print(input)
        hit_detector = False
        distance_at_track_end = None
        energy_at_track_end = None
        point1x = None
        point1y = None
        point1z = None
        point2x = None
        point2y = None
        point2z = None
        

    # print(position, energy_init, theta, phi, charge)
    # hit_detector = False  
    # t1.stop()


    return (hit_detector, distance_at_track_end, energy_at_track_end, 
    energy_init, point1x, point1y, point1z, point2x, point2y, point2z)
