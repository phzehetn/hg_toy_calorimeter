import configparser
import json
import multiprocessing
import os
import sys

import numpy as np
import argh
import time

import ra_pickles
from calo.particle_generator import ParticleGenerator
from calo.calo_generator import CaloV2Generator, CaloV3Generator
from calo.config import set_env, is_laptop

set_env()


def work(simtype, output_path, num_events, num_events_per_file, conf):
    # %%
    import minicalo
    set_env()

    np.random.seed()
    rnd = np.random.randint(0, 1000000000)
    rnd2 = np.random.randint(0, 1000000000)
    rnd3 = np.random.randint(0, 1000000000)
    rnd4 = np.random.randint(0, 1000000000)
    print("Seed is",rnd)


    if 'calo' in conf:
        calo_type = conf['calo']
        if calo_type == 'v2':
            calo_generator = CaloV2Generator()
        elif calo_type == 'v3':
            calo_generator = CaloV3Generator()
        else:
            raise NotImplementedError('Calo not recognised -- it should either be v2 or v3, given calo=',conf['calo'])
    else:
        calo_type = 'v2'
        calo_generator = CaloV2Generator()

    detector_specs = calo_generator.generate()

    nsen = 0
    for layer_upper in detector_specs['layers']:
        for layer in layer_upper:
            for eta_segment in layer['eta_segments']:
                nsen += int(eta_segment['phi_segments'])

    minicalo.initialize(json.dumps(detector_specs), '/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/pythia8306/share/Pythia8'
                        if not is_laptop else '/Users/shahrukhqasim/Workspace/NextCal/miniCalo/pythia8-data',
                        False, rnd, rnd2, rnd3, rnd4) # third argument is collect_full_data
    # particle_pdgid = [11, 22, 211, 111, 15]
    if 'pdgid' in conf:
        particle_pdgid=[int(x) for x in str(conf['pdgid']).split(',')]
    else:
        particle_pdgid = [11, 22, 211, 111]
        if simtype == 'singlepart':
            particle_pdgid += [15]

    range_energy = [0.1, 200]
    if 'energy_min' in conf:
        range_energy = [float(conf['energy_min']), float(conf['energy_max'])]

    range_eta = [3.1, 1.4]
    if 'eta_min' in conf:
        range_eta = [float(conf['eta_min']), float(conf['eta_max'])]

    particle_generator = ParticleGenerator(detector_specs, range_energy=range_energy, particle_pdgid=particle_pdgid, range_eta=range_eta)

    if 'filter_deposited' in conf:
        filter_deposited = conf['filter_deposited'] == 'True' or conf['filter_deposited'] == 'true' or conf['filter_deposited'] == '1'
    else:
        filter_deposited=False

    num_events_written = 0
    while True:
        if num_events_written==num_events:
            break
        simulation_results_array = []
        j = 0
        while True:
            if simtype in {'singlepart', 'singlepart_face'}:
                particle = particle_generator.generate(from_iteraction_point=simtype=='singlepart')
                simulation_result = minicalo.simulate_particle(
                    float(particle['position'][0]),
                    float(particle['position'][1]),
                    float(particle['position'][2]),
                    float(particle['direction'][0]),
                    float(particle['direction'][1]),
                    float(particle['direction'][2]),
                    int(particle['pdgid']),
                    float(particle['energy']),
                )
                simulation_result['simulator'] = calo_type
                if filter_deposited:
                    rechit_sum = np.sum(simulation_result['rechit_energy'])
                    print("Rechit sum", rechit_sum)
                    if rechit_sum < 0.001:
                        continue
                simulation_results_array.append((simulation_result, particle))

            elif simtype=='minbias':
                simulation_result = minicalo.simulate_pu()
                simulation_results_array.append(simulation_result)
            elif simtype=='qqbar2ttbar':
                simulation_result = minicalo.simulate_qqbar2ttbar()
                simulation_results_array.append(simulation_result)
            j += 1

            if j==num_events_per_file:
                break

        print("Put in", output_path)
        dataset = ra_pickles.RandomAccessPicklesWriter(len(simulation_results_array), output_path)
        for x in simulation_results_array:
            dataset.add(x)
            num_events_written += 1
        dataset.close()



def main(simtype, output_path, cores=10, num_simulations=1000000, num_simulations_per_file=100, conf=None):
    """
    Generate a large set of simulations using multiple CPU/CPU cores.

    :param simtype: Type of the simulation. Can be either of the following:
                    1. minbias -- 14 TeV proton-proton interactions
                    2. qqbar2ttbar -- qqbar to ttbar interaction
                    3. singlepart -- generate single particles at the interaction point with 1.4 |eta| < 3.1
                    4. singlepart_face -- generate single particles with 1.4 |eta| < 3.1, 1 cm away from the calorimeter
    :param output-path: A folder where to store the simulations, will be created if it does not exist.
    :param cores: Number of CPU cores
    :param num-simulations: Number of events to generate per CPU core
    :param num-simulations-per-file: Number of events to generate per file per CPU core (just a storage spec)
    """

    os.system('mkdir -p %s'%output_path)

    allowed_sim_types = {'minbias', 'qqbar2ttbar', 'singlepart', 'singlepart_face'}
    if not simtype in allowed_sim_types:
        raise ValueError('Wrong simtype %s: select from '%simtype, allowed_sim_types)

    print("Going to do", simtype, 'and cores', cores)

    processes = []
    for m in range(cores):
        print("Starting")
        p = multiprocessing.Process(target=work, args=(simtype,output_path, num_simulations, num_simulations_per_file, conf))
        p.start()
        time.sleep(0.3)
        processes.append(p)

    for p in processes:
        p.join()

if __name__=='__main__':
    is_config_file = np.any(['.ini' in x for x in sys.argv])
    if is_config_file:
        config_file = sys.argv[1]
        section = sys.argv[2]
        config = configparser.ConfigParser()
        config.read(config_file)

        simtype=config[section]['simtype']
        outpath=config[section]['outpath']
        cores=config[section]['cores']
        num_simulations = config[section]['num-simulations']
        num_simulations_per_file = config[section]['num-simulations-per-file']

        main(simtype, outpath, int(cores), int(num_simulations), int(num_simulations_per_file), config[section])
    else:
        argh.dispatch_command(main)

