
# from EcoMug_pybind11.build import EcoMug as em
import EcoMug as em 
import numpy as np
gen = em.EcoMug()
# gen.SetUseHSphere()  # plane Sphere generation
gen.SetUseSky()  # plane surface generation
gen.SetSkySize((0, 0))  # x and y size of the plane
gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center of the plane
gen.SetSeed(1909)

param = 'std'
param = 'guan'
param = 'gaisser'

angle = 'full'
angle = '30deg'

size = '1e4'
size = '1e5'
size = '3e3'
max_mom = '1e5'
max_mom = '1e6'
max_mom = '1e4'


gen.SetMaximumMomentum(int(float(max_mom)))  # in GeV
if angle=='30deg':
    gen.SetMaximumTheta(np.radians(30))  # in degree
if param=='gaisser':
    gen.SetDifferentialFluxGaisser()
elif param=='guan':
    gen.SetDifferentialFluxGuan()

file_name = f'Em_{param}_{angle}_{size}_xmom{max_mom}.hdf'