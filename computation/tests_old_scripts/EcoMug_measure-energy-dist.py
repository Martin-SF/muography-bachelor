# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from EcoMug.build import EcoMug
import proposal as pp
from numba import vectorize
from numba import jit, njit, prange
import my_py_lib.stopwatch as stopwatch

MU_MINUS_MASS = pp.particle.MuMinusDef().mass

@vectorize(nopython=True)
def change_zenith_convention(angle_in_rad):
    return -angle_in_rad + np.pi


@vectorize(nopython=True)
def calculate_energy_vectorized_GeV(momentum):
    One_momentum_in_MeV = 1000
    return np.sqrt(momentum * momentum +
                        (MU_MINUS_MASS/One_momentum_in_MeV)**2)


@njit()
def JGuan(p, theta):
    A = 0.14*pow(p, -2.7)
    B = 1. / (1 + 1.1*p*np.cos(theta)/115)
    C = 0.054 / (1 + 1.1*p*np.cos(theta)/850)
    return A*(B+C)

gen = EcoMug.EcoMug()
# gen.SetUseHSphere()  # plane Sphere generation
gen.SetUseSky()  # plane surface generation
gen.SetSkySize((0, 0))  # x and y size of the plane
gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center of the plane
gen.SetMinimumMomentum(999)  # in GeV
gen.SetMaximumMomentum(9999)  # in GeV
# gen.SetMaximumTheta(np.radians(30))  # in degree
# gen.SetDifferentialFlux(JGuan)
# gen.SetDifferentialFluxGuan()

STATISTICS = int(1e5) # 1e7:4.5min; 1e6:27s; 2e5:5,4s; 1e4: 0,3s
muon_p = np.zeros(STATISTICS, dtype=float)
muon_theta = np.zeros(STATISTICS, dtype=float)
muon_phi = np.zeros(STATISTICS, dtype=float)
muon_charge = np.zeros(STATISTICS, dtype=int)
muon_e = np.zeros(STATISTICS, dtype=float)


for event in tqdm(range(STATISTICS), disable=False):
    # t1.stop(silent=True)
    # t1.task('generate')  # 60% of time
    gen.Generate()
    # gen.GenerateFromCustomJ()

    # t1.task('writing p')
    muon_p[event] = gen.GetGenerationMomentum()
    # t1.task('writing theta')
    muon_theta[event] = gen.GetGenerationTheta()
    # t1.task('writing phi')
    muon_phi[event] = gen.GetGenerationPhi()
    # t1.task('writing charge')
    muon_charge[event] = gen.GetCharge()

muon_e = calculate_energy_vectorized_GeV(muon_p)  # faster than for loop
muon_theta = change_zenith_convention(muon_theta)

# %%
a = 0
for i in muon_e:
    if i > 2200:
        a +=1
print(a/STATISTICS)