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
    if 'noise_cut' in config[section]:
        cut = np.float64(config[section]['noise_cut'])
    print("Using cut: ", cut)
    include_tracks = True
    if 'include_tracks' in config[section]:
        include_tracks = str2bool(config[section]['include_tracks'])
        print("NOT INCLUDING TRACKS IN THIS SIMULATION!")

    num_cores = 4
    if 'num_event_creation_cores' in config[section]:
        num_cores = int(config[section]['num_event_creation_cores'])

    num_parallel_reading_threads = 20
    if 'num_parallel_reading_threads' in config[section]:
        num_parallel_reading_threads = int(config[section]['num_parallel_reading_threads'])

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

    noise_sigma = 1.2e-5
    if 'noise_sigma' in config[section]:
        noise_sigma = float(config[section]['noise_sigma'])
    noise_mean = 0.0
    if 'noise_mean' in config[section]:
        noise_mean = float(config[section]['noise_mean'])

    sample_isolated_particles = None
    if 'sample_isolated_particles_dist' in config[section]:
        sample_isolated_particles_dist = float(config[section]['sample_isolated_particles_dist'])
        if 'sample_isolated_particles_N' in config[section]:
            sample_isolated_particles_N = lambda: int(config[section]['sample_isolated_particles_N'])
        elif 'sample_isolated_particles_N_mean' in config[section]:
            if all(k in config[section] for k in ['sample_isolated_particles_N_mean',
                                                  'sample_isolated_particles_N_std',
                                                  'sample_isolated_particles_N_min',
                                                  'sample_isolated_particles_N_max']):
                sample_isolated_particles_N_mean = int(config[section]['sample_isolated_particles_N_mean'])
                sample_isolated_particles_N_std = int(config[section]['sample_isolated_particles_N_std'])
                sample_isolated_particles_N_min = int(config[section]['sample_isolated_particles_N_min'])
                sample_isolated_particles_N_max = int(config[section]['sample_isolated_particles_N_max'])

                sample_isolated_particles_N = lambda: min(read_int('sample_isolated_particles_N_max'), max(read_int('sample_isolated_particles_N_min'),
                                                                               int(np.random.normal(
                                                                                   read_int('sample_isolated_particles_N_mean'),
                                                                                   read_int('sample_isolated_particles_N_std')))))

            else:
                raise ValueError("Missing one or more of the following keys in configuration file: "
                                 "'sample_isolated_particles_N_mean', 'sample_isolated_particles_N_std', "
                                 "'sample_isolated_particles_N_min', 'sample_isolated_particles_N_max'")
        else:
            raise ValueError("'sample_isolated_particles_N' must be found in configuration file if "
                             "'sample_isolated_particles_dist' is present")

        sample_isolated_particles = (sample_isolated_particles_N, sample_isolated_particles_dist)

    noise_fluctuations = ('type_a', noise_mean, noise_sigma)

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

    particles_iterator = RandomAccessPicklesReader(particles_folder, error_retry=(0.2, 40))
    pu_iterator = RandomAccessPicklesReader(pu_folder, error_retry=(0.2, 40))


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
                                     num_event_creation_processes=num_cores,
                                     noise_fluctuations=noise_fluctuations,
                                     num_parallel_reading_threads=num_parallel_reading_threads,
                                     include_tracks=include_tracks,
                                     sample_isolated_particles=sample_isolated_particles)
    dataset_creator.process()
