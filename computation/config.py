# set of "tasks" into which the task should be divided. 
# optimal: number of real cores -1
#   (1 core for the dask-scheduler so that it doesn't crash)
N_tasks = 23  #  (phobos=24-1=23) 
N_tasks = 100

hdf_folder = 'data_hdf/'
# hdf_folder = '/scratch/mschoenfeld/data_hdf/'

###########################################################
###########################################################
###########################################################
# EcoMug (d_EM.py and d_EM_lib.py)

# muon parametrisation
param = 'guan'
param = 'std'
param = 'gaisser'  # bachelor thesis results value

# max theta (in degrees)
max_theta = ''  # no limit
max_theta = '30'  # bachelor thesis results value

# amount of muons generated
# 1e7:4.5min; 1e6:27s; 2e5:5,4s; 1e4: 0,3s
STATISTICS = '1e7'  # bachelor thesis results value
STATISTICS = '1e6'
STATISTICS = '1e5'
STATISTICS = '1e4'

# min Energy (in GeV)
min_E = '1e2'
min_E = '1e1'
min_E = '2e2'
min_E = '4e2'
min_E = '5e2'
min_E = '7e2'
min_E = ''  # use standard value
min_E = '6e2'  # bachelor thesis results value

# max Energy (in GeV)
max_E = ''  # use standard value (999 GeV momentum)
max_E = '1e6'  # unuseable with gaisser
max_E = '4e5'  # 30h time for 1e7
max_E = '1e3'  # good for min_E = ''
max_E = '2e5'  # bachelor thesis results value

file_name = f'EcoMug_{STATISTICS}_{param}_min{min_E}_max{max_E}_{max_theta}deg.hdf'


###########################################################
###########################################################
############################################################
# Proposal (d_pp.py and d_pp_lib.py)

# which file to propagate

# file_name = "EcoMug_gaisser_30deg_1e7_min2e2_max2e5.hdf"

# file_name = "EcoMug_gaisser_30deg_1e6_min4e2_max2e5.hdf"

# file_name = "EcoMug_gaisser_30deg_1e7_min5e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_3e7_min5e2_max2e5.hdf" # 1517.6 s

# file_name = "EcoMug_gaisser_30deg_1e2_min6e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e4_min6e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e5_min6e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e6_min6e2_max2e5.hdf"
# file_name = "EcoMug_gaisser_30deg_1e7_min6e2_max2e5.hdf"

# file_name = "EcoMug_gaisser_30deg_23_min6e2_max2e5.hdf"

show_plots = True
if (show_plots):
    show_or_multiplot = 'show'
else:
    show_or_multiplot = ''
print_results = False
silent = True

PP_config = 'sandstein.json'
PP_config = 'sandstein_det.json'
PP_config = 'sandstein_det_genauer_100m_h20.json'
PP_config = 'sandstein_det_genauer.json'
PP_config = 'sandstein_det_genauer_Wasser.json'
PP_config = 'stdrock_perf_tests.json'
PP_config = 'stdrock_perf_tests2ndtiefe.json'
PP_config = 'KH_748m_alt.json'
PP_config = "config_cylinder-huge.json"
PP_config = 'kirchhellen1.json'
PP_config = 'KH_800m.json'


v_cut = ''
v_cut = 0.1
v_cut = 0.0008
v_cut = 0.01
v_cut = 1
v_cut = 0.001  # bachelor thesis results value

multiple_scattering = 'HighlandIntegral'
multiple_scattering = 'Moliere'
multiple_scattering = 'noscattering'
multiple_scattering = 'Highland'  # bachelor thesis results value

pp_config_string = f'vcut={v_cut}_{multiple_scattering}'
file_name_results = f'{PP_config}_{pp_config_string}_{file_name}'



# max_distance, min_energy, hierarchy_condition
propagate_settings = (1e20, 0, 10)  # bachelor thesis results value
# detector_pos = (0, 0, -1205e2)  #old
# da nur z abgefragt wird, ist das nicht der detector 
detector_pos = (0, 0, -1259e2)  # bachelor thesis results value
# if PP_config_file.PP_config == 'kirchhellen1.json':
# else:
#     PP_config_file.detector_pos = (0, 0, -1204.5e2)

# pp_tables_path = "/scratch/mschoenfeld/tables_path"
pp_tables_path = "/tmp"


detector_area = 75  # bachelor thesis results value




path_to_config_file = "config/"+PP_config
import json
# refresh PP_config json with newest v_cut and multiple_scattering
with open(path_to_config_file, 'r+') as file:
    data = json.load(file)
    data['global']['cuts']['v_cut'] = v_cut
    data['global']['scattering']['multiple_scattering'] = multiple_scattering

    file.seek(0)        # <--- should reset file position to the beginning.
    json.dump(data, file, indent=4)
    file.truncate()     # remove remaining part

STATISTICS = int(float(STATISTICS))
