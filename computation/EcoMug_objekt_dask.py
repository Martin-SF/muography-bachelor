
# from EcoMug_pybind11.build import EcoMug as em
import EcoMug as em 
import numpy as np
import py_library.simulate_lib as slib

gen = em.EcoMug()
gen.SetUseSky()  # plane surface generation
gen.SetSkySize((0, 0))  # x and y size of the plane
gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center of the plane
# gen.SetSeed(1909)

param = 'guan'
param = 'std'
param = 'gaisser'

angle = 'full'
angle = '30deg'

size = '3e3'
size = '1e2'
size = '1e3'
size = '1e8'
size = '1e3'
size = '1e8' #WE
size = '1e4'
size = '1e6'
size = '1e5'
size = '1e7' #WE

max_E = '1e5'
max_E = '1e6' # unusabre with gaisser
max_E = '1e4'
max_E = '4e5'  # 30h für 1e7
max_E = '1e3'
max_E = '5e5'
max_E = '1e9'
max_E = '1e7'
max_E = '3e5'  # gut geeingnet als oberes limit für 1e7
max_E = '1e3'  # gut für min-e = 0
max_E = '2e5'  # gut geeingnet als oberes limit für 1e7

min_E = '1e2'
min_E = '1e1'
min_E = ''
min_E = '2e2'
min_E = '7e2'  # gut geeingnet als unteres limit für 1e7
min_E = '6e2'  # gut geeingnet als unteres limit für 1e7

file_name = f'EcoMug_{param}_{angle}_{size}_min{min_E}_max{max_E}.hdf'
if min_E!='':
    gen.SetMinimumMomentum(slib.calculate_momentum_vectorized_GeV(int(float(min_E))))  # in GeV
if max_E!='':
    gen.SetMaximumMomentum(slib.calculate_momentum_vectorized_GeV(int(float(max_E))))  # in GeV
if angle=='30deg':
    gen.SetMaximumTheta(np.radians(30))  # in degree
if param=='gaisser':
    gen.SetDifferentialFluxGaisser()
elif param=='guan':
    gen.SetDifferentialFluxGuan()
pos = [0,0,0]
def Ecomug_generate(i):
    # t1.stop(silent=True)
    # t1.task('generate')  # 60% of time
    gen.Generate()
    # gen.GenerateFromCustomJ()
    # pos = gen.GetGenerationPosition()  # 7 µs
    p = gen.GetGenerationMomentum()
    theta = gen.GetGenerationTheta()
    phi = gen.GetGenerationPhi()
    charge = gen.GetCharge()
    return (pos, p, theta, phi, charge)

