# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import random
import pandas as pd

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14

# %%
from EcoMug.build import EcoMug

gen = EcoMug.EcoMug_Class()
gen.SetUseSky()  # plane surface generation
gen.SetSkySize((10, 10))  # x and y size of the plane
gen.SetSkyCenterPosition((0, 0, 20))  # (x,y,z) position of the center of the plane
gen.SetSeed(1909)
# gen.SetMinimumMomentum(3)  # in GeV
# gen.SetMaximumMomentum(1.5)  # in GeV

# %%
STATISTICS = int(1e6)
muon_position = []
muon_p = []
muon_theta = []
muon_phi = []
muon_charge = []

for event in tqdm(range(STATISTICS), disable=False):
    gen.Generate()

    muon_position.append(gen.GetGenerationPosition())
    muon_p.append(gen.GetGenerationMomentum())
    muon_theta.append(gen.GetGenerationTheta())
    muon_phi.append(gen.GetGenerationPhi())
    muon_charge.append(gen.GetCharge())

# %%
df = pd.DataFrame()
df['position'] = muon_position
df['momentum'] = muon_p
df['theta'] = muon_theta
df['phi'] = muon_phi
df['charge'] = muon_charge

df


