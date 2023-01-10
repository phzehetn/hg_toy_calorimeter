import gzip
import os
import pickle
import sys
import uuid

import numpy as np
from calo.dataset_creator import DatasetCreator
import configparser
from ra_pickles import RandomAccessPicklesWriter, RandomAccessPicklesReader

def str2bool(v):
  return v.lower() in ("yes", "true", "on", "1")

if __name__ == '__main__':
    config_file = sys.argv[1]
    section = sys.argv[2]
    config = configparser.ConfigParser()
    config.read(config_file)

    particles_folder=config[section]['particles_folder']
    pu_folder=config[section]['pu_folder']
    output_folder=config[section]['output_folder']
    num_events_total=int(config[section]['num_events_total'])

    cut = 0.0009

    num_cores = 4
    if 'num_event_creation_cores' in config[section]:
        num_cores = config[section]['num_event_creation_cores']

    if 'calibration_data' in config[section]:
        calib_path = config[section]['calibration_data']
    elif 'HG_TOY_CALORIMETER_CALIBRATION_DATA' in os.environ:
        calib_path = os.getenv('HG_TOY_CALORIMETER_CALIBRATION_DATA')
    else:
        print("Calibration data not found, either set the environmental variable HG_TOY_CALORIMETER_CALIBRATION_DATA"
              " or specify it in the config file.")
        print("The calibration data file can be downloaded from\n"
              "https://github.com/shahrukhqasim/hg_toy_calorimeter_data/raw/master/sensor_data_v3_calibrated.bin")
        raise RuntimeError('Calibration data not found')

    with gzip.open(calib_path, 'rb') as f:
        sensor_data = pickle.load(f)

    read_int = lambda s:int(config[section][s])
    read_float = lambda s:float(config[section][s])
    if 'num_particles' in config[section]:
        num_particles = int(config[section]['num_particles'])
    else:
        num_particles = lambda : min(read_int('num_particles_max'), max(read_int('num_particles_min'), int(np.random.normal(read_int('num_particles_mean'), read_int('num_particles_std')))))

    #num_particles = lambda : min(100, max(10, int(np.random.normal(50,20))))

    num_pu = int(config[section]['num_pu'])

    for i in range(100):
        if type(num_particles) is int:
            print(num_particles, num_pu)
        else:
            print(num_particles(), num_pu)

    # a = input('Should I create this output folder? '+output_folder)
    #
    # if a != 'yes':
    #     print('Okay exiting')
    #     exit(0)

    os.system('mkdir -p %s'%output_folder)
    # print(output_folder)
    # 0/0


    num_events_per_djc = read_int('num_events_per_djc_file') if 'num_events_per_djc_file' in config[section] else min(200, num_events_total)

    pu_phase_cut = read_float('pu_phase_cut') if 'pu_phase_cut' in config[section] else None
    compute_spectators_dist = str2bool(config[section]['compute_spectators_dist']) if 'compute_spectators_dist' in config[section] else True

    # print(pu_phase_cut, num_events_per_djc)
    # 0/0

    particles_iterator = RandomAccessPicklesReader(particles_folder)

    pu_iterator = RandomAccessPicklesReader(pu_folder)

    dataset_creator = DatasetCreator(pu_iterator=pu_iterator,
                                     particles_iterator=particles_iterator,
                                     output_path=output_folder,
                                     rechit_cut=cut,
                                     sensor_data=sensor_data,
                                     num_events_per_djc_file=num_events_per_djc,
                                     num_particles_per_event=num_particles,
                                     num_pu_per_event=num_pu,
                                     num_events_total=num_events_total,
                                     pu_phase_cut=pu_phase_cut,
                                     min_hits_cut=1, compute_spectators_dist=compute_spectators_dist,
                                     num_event_creation_processes=num_cores)
    dataset_creator.process()
