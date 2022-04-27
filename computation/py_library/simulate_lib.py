
import pandas as pd
import numpy as np
from numba import vectorize
import proposal as pp

FLOAT_TYPE = np.float64
MU_MINUS_MASS_squared_GeV = (pp.particle.MuMinusDef().mass/1000)**2

def transform_position_list(arr):
    len_arr = len(arr)
    a = np.zeros(shape=(len_arr, 3), dtype=FLOAT_TYPE)
    for i in range(len_arr):
        a[i] = np.array(arr[i], dtype=FLOAT_TYPE)
    return a


def transform_position_posxyz(x, y, z):
    a = np.zeros(shape=(len(x), 3), dtype=float)
    a[:, 0] = x
    a[:, 1] = y
    a[:, 2] = z
    return a


def read_muon_data(file_name, key):
    data = pd.read_hdf(file_name, key=key)
    # if 'data' not in locals():
    #     data = pd.read_hdf(file_name, key=f'muons_{size}')

    # t.task('transform_positioning position data')
    # data_position = transform_position_list(data['position'])
    data_position = transform_position_posxyz(
        data['pos_x'], data['pos_y'], data['pos_z']
        )

    data_momentum = np.array(data['momentum'], dtype=FLOAT_TYPE)
    data_energy = np.array(data['energy'], dtype=FLOAT_TYPE)
    data_theta = np.array(data['theta'], dtype=FLOAT_TYPE)
    data_phi = np.array(data['phi'], dtype=FLOAT_TYPE)
    data_charge = np.array(data['charge'], dtype=np.int8)

    return (data_position, data_momentum, data_energy,
             data_theta, data_phi, data_charge)


def which_size(a):
    if a<=int(1e4):
        return {'npartitions' : 100}
    if a<=int(1e5):
        return {'npartitions' : 500}
    if a<=int(1e6):
        return {'npartitions' : 1000}
    if a<=int(1e7) or a>=int(1e7):
        return {'npartitions' : 10000}


@vectorize(nopython=True)
def change_zenith_convention(angle_in_rad):
    return -angle_in_rad + np.pi


# calculate energy from momentum, expecting GeV, 
# calculating MU_MINUS_MASS to GeV with One_momentum_in_MeV
# expecting momentum in GeV, output E in GeV (muon mass in GeV)
@vectorize(nopython=True)
def calculate_energy_vectorized_GeV(momentum):
    return np.sqrt(momentum * momentum + MU_MINUS_MASS_squared_GeV)