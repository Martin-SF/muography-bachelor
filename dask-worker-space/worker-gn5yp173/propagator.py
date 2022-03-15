#%%

import proposal as pp
import os, sys
# os.chdir(os.path.dirname(__file__))

pp.InterpolationSettings.tables_path = "/tmp"
config = "config_cylinder-huge.json"
config = "configs/"+config

prop_plus = pp.Propagator(
                particle_def=pp.particle.MuPlusDef(),
                path_to_config_file=config
        )

prop_minus = pp.Propagator(
    particle_def=pp.particle.MuMinusDef(),
    path_to_config_file=config
)

test = 5