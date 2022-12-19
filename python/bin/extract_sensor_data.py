import gzip
import pickle
import sys
import numpy as np
import json
from calo.calo_generator import CaloV3Generator
from calo.config import set_env, is_laptop



set_env()
import minicalo

calo_generator = CaloV3Generator()
detector_specs = calo_generator.generate()
minicalo.initialize(json.dumps(detector_specs),
                    '/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/pythia8306/share/Pythia8'
                    if not is_laptop else '/Users/shahrukhqasim/Workspace/NextCal/miniCalo/pythia8-data',
                    False, 1, 2, 3, 4)  # third argument is collect_full_data

sensor_data = minicalo.get_sensor_data()


with gzip.open(sys.argv[1], 'wb') as f:
    pickle.dump(sensor_data, f)
