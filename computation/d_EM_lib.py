# from EcoMug_pybind11.build import EcoMug as em
import EcoMug as em  # findable when setting PYTHONPATH to build folder
import numpy as np
from numba import vectorize
# import py_library.simulate_lib as slib  # this import caused problems on some systems
import config as config_file
import proposal as pp
from importlib import reload
reload(config_file)

MU_MINUS_MASS_squared_GeV = (pp.particle.MuMinusDef().mass/1000)**2

# calculating momentum of muon for given energy
@vectorize(nopython=True)
def calculate_momentum_vectorized_GeV(energy): 
    return np.sqrt(energy * energy - MU_MINUS_MASS_squared_GeV)

gen = em.EcoMug()
gen.SetUseSky()  # plane surface generation
gen.SetSkySize((0, 0))  # x and y size of the plane
gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center of the plane
gen.SetSeed(int(np.random.random()*1000))


if config_file.min_E!='':
    gen.SetMinimumMomentum(calculate_momentum_vectorized_GeV(int(float(config_file.min_E))))
if config_file.max_E!='':
    gen.SetMaximumMomentum(calculate_momentum_vectorized_GeV(int(float(config_file.max_E))))
if config_file.max_theta!='':
    gen.SetMaximumTheta(np.radians(int(config_file.max_theta)))  # in degrees
if config_file.param=='gaisser':
    gen.SetDifferentialFluxGaisser()
elif config_file.param=='guan':
    gen.SetDifferentialFluxGuan()

# pos = [0,0,0]
pos = 0
def Ecomug_generate(i):
    if config_file.param=='std':
        gen.Generate()
    else:
        gen.GenerateFromCustomJ()
    
    # pos = gen.GetGenerationPosition()  # 7 µs
    p = gen.GetGenerationMomentum()
    theta = gen.GetGenerationTheta()
    phi = gen.GetGenerationPhi()
    charge = gen.GetCharge()
    return (pos, p, theta, phi, charge)
