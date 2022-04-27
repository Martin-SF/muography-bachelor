
# from EcoMug_pybind11.build import EcoMug as em
import EcoMug as em 
import numpy as np

gen = em.EcoMug()
gen.SetUseSky()  # plane surface generation
gen.SetSkySize((0, 0))  # x and y size of the plane
gen.SetSkyCenterPosition((0, 0, 0))  # (x,y,z) position of the center of the plane
# gen.SetSeed(1909)

param = 'std'
param = 'guan'
param = 'gaisser'

angle = 'full'
angle = '30deg'

size = '3e3'
size = '1e2'
size = '1e3'
size = '1e8'
size = '1e3'
size = '1e4'
size = '1e5'
size = '1e7' #WE
size = '1e6'
max_mom = '1e5'
max_mom = '1e6' # unusabre with gaisser
max_mom = '1e4'
max_mom = '4e5'  # 30h für 1e7
max_mom = '1e3'
max_mom = '5e5'
max_mom = '1e9'
max_mom = '1e7'
max_mom = '3e5'  # gut geeingnet als oberes limit für 1e7

min_mom = '1e2'
min_mom = '1e1'
min_mom = '1e2'
min_mom = ''
min_mom = '5e2'  # gut geeingnet als unteres limit für 1e7

file_name = f'EcoMug_{param}_{angle}_{size}_min{min_mom}_max{max_mom}.hdf'

if min_mom!='':
    gen.SetMinimumMomentum(int(float(min_mom)))  # in GeV
gen.SetMaximumMomentum(int(float(max_mom)))  # in GeV
if angle=='30deg':
    gen.SetMaximumTheta(np.radians(30))  # in degree
if param=='gaisser':
    gen.SetDifferentialFluxGaisser()
elif param=='guan':
    gen.SetDifferentialFluxGuan()

def Ecomug_generate(i):
    # t1.stop(silent=True)
    # t1.task('generate')  # 60% of time
    # genGenerate()gen.GenerateFromCustomJ()
    gen.GenerateFromCustomJ()
    pos = gen.GetGenerationPosition()  # 7 µs
    p = gen.GetGenerationMomentum()
    theta = gen.GetGenerationTheta()
    phi = gen.GetGenerationPhi()
    charge = gen.GetCharge()
    return (pos, p, theta, phi, charge)

