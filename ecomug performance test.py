# %%
# from turtle import color
import numpy as np
from tqdm import tqdm
import pandas as pd
# from scipy.stats import norm
import proposal as pp
from EcoMug.build import EcoMug
from numba import vectorize
import my_py_lib.stopwatch as stopwatch
from importlib import reload
reload(stopwatch)


MU_MINUS_MASS = pp.particle.MuMinusDef().mass


@vectorize(nopython=True)
def change_azimuth_convention(angle_in_rad):
    return -angle_in_rad + np.pi


@vectorize(nopython=True)
def calculate_energy_vectorized_GeV(momentum):
    One_momentum_in_MeV = 1000
    return np.sqrt(momentum * momentum +
                        (MU_MINUS_MASS/One_momentum_in_MeV)**2)


###########  GERERATING WITH ECOMUG
############################################################
############################################################
############################################################
############################################################
t = stopwatch.stopwatch(
    title='generating ecomug muons', selfexecutiontime_micros=0,
    time_unit='ms'
)
t.task('generating ecomug object')
gen = EcoMug.EcoMug()
gen.SetUseHSphere()  # plane Sphere generation
gen.SetSeed(1909)
file_name = "EcoMug_fullspectrum.hdf"
file_name = "EcoMug_test_new_position.hdf"
STATISTICS = int(1e7)

# t.task('generating arrays')
t.task('generating arrays + generating', True)
# muon_pos = [None]*STATISTICS  # 2,13 - 2,24 s
# muon_pos = np.zeros(STATISTICS*3, dtype=float).reshape(STATISTICS, 3)
muon_pos = np.zeros(shape=(STATISTICS, 3), dtype=float)
# muon_pos = [([float]*3)]*STATISTICS  # same as [None]
# muon_pos = np.empty(STATISTICS*3).reshape(STATISTICS, 3)  dataframe doesnt take array as pos, need to take a len(list)=3 list for storing it as object 

# muon_theta = [float]*STATISTICS
# muon_phi = [float]*STATISTICS
# muon_charge = [int]*STATISTICS
# muon_e = [float]*STATISTICS
# muon_p = [float]*STATISTICS

muon_p = np.zeros(STATISTICS, dtype=float)
muon_theta = np.zeros(STATISTICS, dtype=float)
muon_phi = np.zeros(STATISTICS, dtype=float)
muon_charge = np.zeros(STATISTICS, dtype=int)
muon_e = np.zeros(STATISTICS, dtype=float)


# t.task('generating muons')
# = np.zeros(STATISTICS)  # 5 % slower
# = []  # about as fast as preallocated

# t1 = stopwatch.stopwatch(title = 'generating ecomug muons', selfexecutiontime_in_ms=1.4, time_unit='µs')
# t1.task()
for event in tqdm(range(STATISTICS), disable=False):
    # t1.stop(silent=True)
    # t1.task('generate')  # 60% of time
    gen.Generate()
    # t1.task('writing data')  # 40% of time
    # t1.task('writing pos')
    # muon_pos[event]     = np.array(gen.GetGenerationPosition())  # 12 µs
    muon_pos[event] = gen.GetGenerationPosition()  # 7 µs
    # t1.task('writing p')
    muon_p[event] = gen.GetGenerationMomentum()
    # t1.task('writing theta')
    muon_theta[event] = gen.GetGenerationTheta()
    # t1.task('writing phi')
    muon_phi[event] = gen.GetGenerationPhi()
    # t1.task('writing charge')
    muon_charge[event] = gen.GetCharge()
    # if (event==STATISTICS-1):
    #     t1.stop()


t.task('calculation energy')
t.stop(silent=True)
quit()
#%%
muon_e = calculate_energy_vectorized_GeV(muon_p)  # faster than for loop

t.task('write to df')
df = pd.DataFrame()
df['pos_x'] = muon_pos[:, 0]
df['pos_y'] = muon_pos[:, 1]
df['pos_z'] = muon_pos[:, 2]
# df['position'] = muon_pos
df['momentum'] = muon_p
df['energy'] = muon_e
df['theta'] = muon_theta
df['phi'] = muon_phi
df['charge'] = muon_charge

t.task('write to HDF file')
df.to_hdf(file_name, key=f'muons_{STATISTICS}')
# t.stop(silent=False)
t.stop(silent=True)


# ---generating ecomug muons on phobos---
# +--------------------------+----------+------------+
# |           task           | duration | % of TOTAL |
# +--------------------------+----------+------------+
# | generating ecomug object |  0.4 ms  |    0.1 %   |
# |    generating arrays     |  0.9 ms  |    0.2 %   |
# |     generating muons     | 510.6 ms |   84.3 %   |
# |    calculation energy    | 71.5 ms  |   11.8 %   |
# |       write to df        |  4.8 ms  |    0.8 %   |
# |    write to HDF file     | 17.7 ms  |    2.9 %   |
# |            --            |    --    |     --     |
# |          TOTAL           | 605.9 ms |  100.0 %   |
# +--------------------------+----------+------------+
