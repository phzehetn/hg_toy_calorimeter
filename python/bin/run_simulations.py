import json
import multiprocessing
import numpy as np
import argh
import time

import ra_pickles
from calo.particle_generator import ParticleGenerator
from calo.calo_generator import CaloV2Generator
from calo.config import set_env, is_laptop

set_env()


def work(simtype):
    # %%
    import minicalo
    set_env()

    # if not part:
    #     raise RuntimeError('Not implemented')
    # output_path = '/eos/home-s/sqasim/Datasets/b4toys/run4/pu2' if not is_laptop else '../sample_pu/bid_f'
    # output_path = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/b4toys/run2/pu_raw/v1' if not is_laptop else '../sample_pu/bid_f'
    output_path = '/eos/user/s/sqasim/Datasets/b4toys/run2/pu_raw/v1' if not is_laptop else '../sample_pu/with_fixed_kenergy'

    if simtype=='singlepart':
        output_path = '/eos/user/s/sqasim/Datasets/b4toys/run2/particles_raw/v1' if not is_laptop else '../sample_particles/v1'
    elif simtype=='singlepart_face':
        output_path = '/eos/user/s/sqasim/Datasets/b4toys/run2/particles_raw/v1face' if not is_laptop else '../sample_particles/v1t'
    elif simtype=='minbias':
        output_path = '/eos/user/s/sqasim/Datasets/b4toys/run2/pu_raw/v1' if not is_laptop else '../sample_pu/1'
    elif simtype=='qqbar2ttbar':
        output_path = '/eos/user/s/sqasim/Datasets/b4toys/run2/qqbar2ttbar_raw/v1' if not is_laptop else '../sample_qqbar2ttbar/3'


    np.random.seed()
    rnd = np.random.randint(0, 1000000000)
    rnd2 = np.random.randint(0, 1000000000)
    rnd3 = np.random.randint(0, 1000000000)
    rnd4 = np.random.randint(0, 1000000000)
    print("Seed is",rnd)

    detector_specs_2 = CaloV2Generator().generate()

    minicalo.initialize(json.dumps(detector_specs_2), '/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/pythia8306/share/Pythia8'
                        if not is_laptop else '/Users/shahrukhqasim/Workspace/NextCal/miniCalo/pythia8-data',
                        False, rnd, rnd2, rnd3, rnd4) # third argument is collect_full_data
    # particle_pdgid = [11, 22, 211, 111, 15]

    particle_pdgid = [11, 22, 211, 111]
    if simtype == 'singlepart':
        particle_pdgid += [15]
    particle_generator = ParticleGenerator(detector_specs_2, range_energy=[0.1, 200], particle_pdgid=particle_pdgid)
    for i in range(400):
        simulation_results_array = []
        for j in range(5 if is_laptop else 100):
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
                simulation_results_array.append((simulation_result, particle))
            elif simtype=='minbias':
                simulation_result = minicalo.simulate_pu()
                simulation_results_array.append(simulation_result)
            elif simtype=='qqbar2ttbar':
                simulation_result = minicalo.simulate_qqbar2ttbar()
                simulation_results_array.append(simulation_result)


        print("Put in", output_path)
        dataset = ra_pickles.RandomAccessPicklesWriter(len(simulation_results_array), output_path)
        for x in simulation_results_array:
            dataset.add(x)
        dataset.close()



def main(simtype, cores=10):
    allowed_sim_types = {'minbias', 'qqbar2ttbar', 'multipart', 'singlepart', 'singlepart_face'}
    if not simtype in allowed_sim_types:
        raise ValueError('Wrong simtype %s: select from '%simtype, allowed_sim_types)

    print("Going to do", simtype, 'and cores', cores)

    processes = []
    for m in range(cores):
        print("Starting")
        p = multiprocessing.Process(target=work, args=(simtype,))
        p.start()
        time.sleep(0.3)
        processes.append(p)

    for p in processes:
        p.join()

if __name__=='__main__':
    argh.dispatch_command(main)

