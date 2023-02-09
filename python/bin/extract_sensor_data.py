import gzip
import pickle
import sys

import argh as argh
import numpy as np
import json
from calo.calo_generator import CaloV3Generator, CaloV4Generator
from calo.config import set_env, is_laptop

def main(output_file, calo_type='v3'):
    set_env()
    import minicalo

    if calo_type == 'v3':
        CGenerator = CaloV3Generator
    elif calo_type == 'v4':
        CGenerator = CaloV4Generator
    else:
        raise NotImplementedError('Error')

    calo_generator = CGenerator()

    detector_specs = calo_generator.generate()
    minicalo.initialize(json.dumps(detector_specs),
                        '/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/pythia8306/share/Pythia8'
                        if not is_laptop else '/Users/shahrukhqasim/Workspace/NextCal/miniCalo/pythia8-data',
                        False, 1, 2, 3, 4)  # third argument is collect_full_data

    sensor_data = minicalo.get_sensor_data()

    with gzip.open(output_file, 'wb') as f:
        pickle.dump(sensor_data, f)

    print("Sensor data saved to %s for %s calorimeter."%(output_file, calo_type))

argh.dispatch_command(main)