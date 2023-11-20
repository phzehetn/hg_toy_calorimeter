import gzip
import io
import os
import pickle
import threading
import time
import uuid
import queue

import numpy as np
import multiprocessing

from ra_pickles import RandomAccessPicklesWriter
from calo.detector_math import x_y_z_to_eta_phi_theta
from calo.config import is_laptop
from calo.event_generator import EventGenerator

if not is_laptop:
    from datastructures.TrainData_NanoML import TrainData_NanoML, find_pcas
    from DeepJetCore import SimpleArray

def unprocess(data):
    binary_data = io.BytesIO(data)
    gzipfile = gzip.GzipFile(fileobj=binary_data, mode='rb')
    data_loaded = pickle.load(gzipfile)
    gzipfile.close()
    return data_loaded


def grow():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def event_making_process(
        q_in, q_out, sensor_data,
        rechit_cut, min_hits_cut=3,
        pu_phase_cut=None, pu_eta_cut=None,
        compute_spectators_dist=True,
        noise_fluctuations=('type_a', 0, 5e-6),
        include_tracks=True,
        sample_isolated_particles=None,
        alter_muon_truth=False):


    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    grow()
    np.random.seed()

    gen = EventGenerator(
        sensor_data,
        noise_fluctuations=noise_fluctuations,
        cut=rechit_cut,
        num_hits_cut=min_hits_cut,
        reduce=True,
        area_normed_cut=True,
        include_tracks=include_tracks,
        sample_isolated_particles=sample_isolated_particles,
        alter_muon_truth=alter_muon_truth)

    while True:
        try:
            data = q_in.get(timeout=3)
            if data is None:
                print("Getting nones, exiting...")
                break
        except queue.Empty:
            continue
        t1 = time.time()
        particles, pu = data
        try:
            # particles = [unprocess(x) for x in particles]
            # pu = [unprocess(x) for x in pu]
            pass
        # except OSError:
        #     print("Error occurred... handled?")
        #     continue
        except:
            # TODO: Do better exception handling. This is too general.
            print("Error occurred... handled?")
            continue
        particles = [x[0] if type(x) is tuple else x for x in particles]

        if len(particles) + len(pu) == 0:
            continue
        if len(pu) == 0:
            if sum([sum(particles[x]['particles_total_energy_deposited_all']) for x in range(len(particles))]) == 0:
                print("Continuing")
                continue
        gen.add(particles, minbias=False)
        if len(pu) > 0:
            gen.add(pu, phase_cut=pu_phase_cut, eta_cut=pu_eta_cut, minbias=True)
        result = gen.process()

        # print("XYZ num particle events", len(simulations))
        # result_func = combine_events(None, simulations, sensor_data, noise_fluctuations=noise_fluctuations,
        #                         num_hits_cut=3, cut=rechit_cut, reduce=True)
        # print("XYZ Num hits", len(result['rechit_energy']), len(result_func['rechit_energy']))
        print("Event processing xyz done in", time.time() - t1, "seconds")

        t1 = time.time()
        feat, truth = get_djc_data(result, compute_spectators_dist=compute_spectators_dist)
        unique, counts = np.unique(truth['t_idx'], return_counts=True)
        counts = counts[unique >= 0]
        if len(counts) == 0:
            continue
        print("Shower classes", np.unique(result['truth_assignment_shower_class'], return_counts=True))
        print("DJC conversion with %d hits done in with min hits %d" % (len(feat), np.min(counts)), time.time() - t1,
              "seconds")

        q_out.put((feat, truth))


def get_djc_data(result, compute_spectators_dist=True):
    recHitEnergy = result['rechit_energy']
    recHitTrack = result['rechit_is_track']
    recHitX = result['rechit_x']
    recHitY = result['rechit_y']
    recHitZ = result['rechit_z']
    recHitTime = recHitEnergy * 0
    recHitHitR = np.sqrt(result['rechit_area'])

    recHitEta, _, recHitTheta = x_y_z_to_eta_phi_theta(recHitX, recHitY, recHitZ)
    recHitR = np.sqrt(recHitX ** 2 + recHitY ** 2 + recHitZ ** 2)

    features = np.concatenate([
        recHitEnergy[..., np.newaxis],
        recHitEta[..., np.newaxis],
        recHitTrack[..., np.newaxis],  # indicator if it is track or not
        recHitTheta[..., np.newaxis],
        recHitR[..., np.newaxis],
        recHitX[..., np.newaxis],
        recHitY[..., np.newaxis],
        recHitZ[..., np.newaxis],
        recHitTime[..., np.newaxis],
        recHitHitR[..., np.newaxis],
    ], axis=-1)
    features = features.astype(np.float32)

    unique_showers, unique_shower_indices = np.unique(result['truth_assignment'], return_index=True)
    unique_shower_indices = unique_shower_indices[unique_showers != -1]
    unique_showers = unique_showers[unique_showers != -1]

    t_is_unique = result['truth_assignment'] * 0
    t_is_unique[unique_shower_indices] = 1

    spectator_dist = result['truth_assignment_energy'] * 0

    if compute_spectators_dist:
        for u in unique_showers:
            f = result['truth_assignment'] == u
            X = result['rechit_x'][f][..., np.newaxis]
            Y = result['rechit_y'][f][..., np.newaxis]
            Z = result['rechit_z'][f][..., np.newaxis]
            x_to_fit = np.concatenate((X, Y, Z), axis=-1)
            spectators_shower_dist = find_pcas(x_to_fit, PCA_n=2, min_hits=10)

            if spectators_shower_dist is None:
                spectators_shower_dist = np.zeros(len(X), np.float)

            spectator_dist[np.argwhere(f)[:, 0]] = spectators_shower_dist
    else:
        print("Not computing spectator distance...")

    truth = {}
    truth['t_idx'] = result['truth_assignment'][..., np.newaxis]
    truth['t_energy'] = result['truth_assignment_energy'][..., np.newaxis]
    truth['t_pos'] = np.concatenate((result['truth_assignment_x'][..., np.newaxis],
                                     result['truth_assignment_y'][..., np.newaxis],
                                     result['truth_assignment_z'][..., np.newaxis]), axis=-1)
    truth['t_shower_class'] = result['truth_assignment_shower_class'][..., np.newaxis]
    truth['t_only_minbias'] = result['truth_assignment_only_minbias'][..., np.newaxis]
    truth['t_time'] = truth['t_energy'] * 0
    truth['t_pid'] = result['truth_assignment_pdgid'][..., np.newaxis]
    truth['t_spectator'] = spectator_dist[..., np.newaxis]
    truth['t_fully_contained'] = truth['t_idx'] * 0 + 1
    truth['t_rec_energy'] = result['truth_assignment_energy_dep'][..., np.newaxis]
    truth['t_is_unique'] = t_is_unique[..., np.newaxis]

    return features, truth


class DatasetCreator():
    def __init__(self, pu_iterator,
                 particles_iterator,
                 output_path,
                 rechit_cut,
                 sensor_data,
                 num_events_per_djc_file,
                 num_particles_per_event,
                 num_pu_per_event,
                 num_events_total,
                 pu_phase_cut,
                 min_hits_cut=3,
                 compute_spectators_dist=True,
                 noise_fluctuations=('type_a', 0,1.2e-5),
                 num_event_creation_processes=4,
                 num_parallel_reading_threads=20,
                 include_tracks=True,
                 sample_isolated_particles=None,
                 alter_muon_truth=False):
        self.output_path = output_path
        self.rechit_cut = rechit_cut
        self.sensor_data = sensor_data
        self.num_events_per_djc_file = num_events_per_djc_file
        self.num_events_total = num_events_total
        self.min_hits_cut = min_hits_cut
        self.last_file_writing_index = 0
        self.compute_spectators_dist = compute_spectators_dist
        self.num_event_creation_processes = num_event_creation_processes

        f_num_particles_per_event = num_particles_per_event
        f_num_pu_per_event = num_pu_per_event
        if type(num_particles_per_event) is int:
            f_num_particles_per_event = lambda: num_particles_per_event
        if type(num_pu_per_event) is int:
            f_num_pu_per_event = lambda: num_pu_per_event

        self.num_particles_per_event = f_num_particles_per_event
        self.num_pu_per_event = f_num_pu_per_event

        self.pu_phase_cut = pu_phase_cut

        self.pu_iterator = pu_iterator
        self.particles_iterator = particles_iterator

        self.rebuild_pu_samples()
        self.rebuild_part_samples()
        self.noise_fluctuations = noise_fluctuations 
        self.include_tracks = include_tracks

        self.sample_isolated_particles = sample_isolated_particles
        self.alter_muon_truth = alter_muon_truth

        self.num_parallel_reading_threads = num_parallel_reading_threads

    def data_loading_thread(self):
        n_loaded = 0
        while n_loaded < self.num_events_total:
            # Don't load too much and hog the memory -- wait if too much writing or event creation is in process
            # TODO: This is a possible memory leak if writing is the bottleneck. For now its okay.
            # (writing is a not a bottleneck for now)
            while (self.events_input_cache.qsize()) > 30:
                time.sleep(0.05)

            num_part = self.num_particles_per_event()
            num_pu = self.num_pu_per_event()

            print("Remaining pu", len(self.pu_samples), "and part", len(self.particles_samples))
            if len(self.pu_samples) < num_pu:
                self.rebuild_pu_samples()

            if len(self.particles_samples) < num_part:
                self.rebuild_part_samples()

            load_particle_samples = self.particles_samples[0:num_part]
            self.particles_samples = self.particles_samples[num_part:]

            load_pu_samples = self.pu_samples[0:num_pu]
            self.pu_samples = self.pu_samples[num_pu:]

            try:
                particles = self.load_n(self.particles_iterator, load_particle_samples)
                pu = self.load_n(self.pu_iterator, load_pu_samples)
            except OSError:
                print("Error -- handled?")
                continue

            if len(particles) != num_part or len(pu) != num_pu:
                raise RuntimeError('Error occurred, more samples/event than in the dataset?')

            self.events_input_cache.put((particles, pu))
            n_loaded += 1

        print("Loaded all requests -- now put in some nones")

        for i in range(20*self.num_event_creation_processes):
            self.events_input_cache.put(None)

            # if self.events_cache.qsize() > self.num_events_per_djc_file:
            #     self.create_djc_file()


        # if self.events_cache.qsize() > 0:
        #     self.create_djc_file()

    def rebuild_pu_samples(self, remaining=[]):
        print("\n\n\nBuilding and shuffling pu samples\n\n\n")
        all_samples = set(
            np.arange(self.pu_iterator.get_total()) if self.pu_iterator is not None else np.array([], np.int).tolist())
        all_samples = list(all_samples - set(remaining))
        np.random.shuffle(all_samples)
        self.pu_samples = remaining + all_samples

    def rebuild_part_samples(self, remaining=[]):
        print("\n\n\nBuilding and shuffling part samples\n\n\n")
        all_samples = set(
            np.arange(self.particles_iterator.get_total()) if self.particles_iterator is not None else np.array([],
                                                                                                                np.int).tolist())
        all_samples = list(all_samples - set(remaining))
        np.random.shuffle(all_samples)
        self.particles_samples = remaining + all_samples

    def load_n(self, reader, samples):
        if len(samples) == 0:
            return []

        t1 = time.time()
        data = list(reader.get_multi_in_parallel(samples, timeout=120))

        error_occurred_in_reading = np.any([x is None for x in data])
        if error_occurred_in_reading:
            raise OSError("Error reading")

        print("Loaded in", time.time() - t1)
        return data

    def write_out(self, data):
        if data.qsize() == 0:
            return

        if not is_laptop:
            rs = [0]
            Fs = []
            Ts = []
            total = 0
            while data.qsize() > 0:
                feat, truth = data.get()
                Fs.append(feat)
                Ts.append(truth)
                total += len(feat)
                rs.append(total)

            rs = np.array(rs, np.int64)
            F_full = np.concatenate(Fs, axis=0)
            T_full = dict()

            keys = Ts[0].keys()
            for k in keys:
                T_full[k] = np.concatenate([Ts[i][k] for i in range(len(Ts))])

            if not is_laptop:
                rechit_features = [SimpleArray(F_full, rs, name="recHitFeatures")]
                truth_all = []
                truth_all += [SimpleArray(T_full['t_idx'].astype(np.int32), rs, name='t_idx')]
                truth_all += [SimpleArray(T_full['t_energy'].astype(np.float32), rs, name='t_energy')]
                truth_all += [SimpleArray(T_full['t_pos'].astype(np.float32), rs, name='t_pos')]
                truth_all += [SimpleArray(T_full['t_time'].astype(np.float32), rs, name='t_time')]
                truth_all += [SimpleArray(T_full['t_pid'].astype(np.float32), rs, name='t_pid')]
                truth_all += [SimpleArray(T_full['t_spectator'].astype(np.float32), rs, name='t_spectator')]
                truth_all += [SimpleArray(T_full['t_fully_contained'].astype(np.float32), rs, name='t_fully_contained')]
                truth_all += [SimpleArray(T_full['t_rec_energy'].astype(np.float32), rs, name='t_rec_energy')]
                truth_all += [SimpleArray(T_full['t_is_unique'].astype(np.int32), rs, name='t_is_unique')]
                truth_all += [SimpleArray(T_full['t_only_minbias'].astype(np.int32), rs, name='t_only_minbias')]
                truth_all += [SimpleArray(T_full['t_shower_class'].astype(np.int32), rs, name='t_shower_class')]

                print(F_full.shape, T_full['t_idx'].shape)
                x = rechit_features + truth_all
                y = []
                z = []
                traindata = TrainData_NanoML()
                traindata._store(x, y, z)
                #
                # file = str(uuid.uuid4()) + '.djctd'
                file = '%05d' % self.last_file_writing_index + '.djctd'
                self.last_file_writing_index += 1
                path = os.path.join(self.output_path, file)
                with open(path, 'wb') as f:
                    pickle.dump([12, 3], f)
                print("Writing djc to", path)
                traindata.writeToFile(path)

        else:
            dataset = RandomAccessPicklesWriter(data.qsize(),
                                                  '/Users/shahrukhqasim/Workspace/NextCal/ShahRukhStudies/toydetector2/scripts/sample_events_multipart')
            while data.qsize() > 0:
                dataset.add(data.get())
            dataset.close()

    def data_writing_thread(self):
        num_written = 0
        data_queue = queue.Queue()
        while True:
            try:
                data = self.events_output_cache.get(timeout=3)
                if data is None:
                    self.write_out(data_queue)
                    break
                data_queue.put(data)
                if data_queue.qsize() >= self.num_events_per_djc_file:
                    self.write_out(data_queue)

                num_written += 1

                print("\n\nWritten data...", num_written, '\n\n')
            except queue.Empty:
                continue
            except Exception as e:
                print(e)
                print("This error occurred... continuing so writing thread does not fail but check.")
                continue

    def process(self):
        print("Starting processing...")
        if self.particles_iterator is not None:
            self.particles_iterator.start_parallel_retrieval_threads(self.num_parallel_reading_threads)
        if self.pu_iterator is not None:
            self.pu_iterator.start_parallel_retrieval_threads(self.num_parallel_reading_threads)

        self.loading_thread = threading.Thread(target=self.data_loading_thread, args=())
        self.writing_thread = threading.Thread(target=self.data_writing_thread, args=())
        self.events_input_cache = (multiprocessing).Queue()
        self.events_output_cache = (multiprocessing).Queue()

        # Start processes which create events in parallel
        self.event_creator_processes = []
        for i in range(self.num_event_creation_processes):
            p = multiprocessing.Process(
		target=event_making_process, args=(
			self.events_input_cache, 
			self.events_output_cache, 
			self.sensor_data, 
			self.rechit_cut, 
			self.min_hits_cut, 
			self.pu_phase_cut, 
			None, 
			self.compute_spectators_dist, 
			self.noise_fluctuations,
			self.include_tracks,
                        self.sample_isolated_particles,
                        self.alter_muon_truth,
			)
		)

            p.start()
            self.event_creator_processes.append(p)

        self.loading_thread.start()
        self.writing_thread.start()

        for p in self.event_creator_processes:
            p.join()

        self.events_output_cache.put(None)

        self.loading_thread.join()
        self.writing_thread.join()

        if self.particles_iterator is not None:
            self.particles_iterator.close_parallel_retrieval_threads()
        if self.pu_iterator is not None:
            self.pu_iterator.close_parallel_retrieval_threads()
