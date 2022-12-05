import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import connected_components
import detector_math as dm

# import experiment_database_manager as edm
# import sql_credentials

# db = edm.ExperimentDatabaseManager(mysql_credentials=sql_credentials.credentials)
# db.set_experiment('event_merging_checks_v1')
from toydetector2.modules.merging_ops import merge_hits
from numba import njit
import tensorflow as tf


@njit
def _get_particle_that_max_dep_on_sensor(particle_id, sensor_idx, deposits, max_parts_result):
    # sensor_idx has to be sorted
    particle_max = particle_id[0]
    deposit_max = deposits[0]
    sensor_tracking = sensor_idx[0]

    for i in range(len(deposits)):
        if i != 0:
            if sensor_idx[i] < sensor_idx[i - 1]:
                raise RuntimeError('Alignment issue')

        if not (sensor_idx[i] == sensor_tracking):
            # Accumulate data
            max_parts_result[sensor_tracking] = particle_max

            # Reset state
            sensor_tracking = sensor_idx[i]
            particle_max = particle_id[i]
            deposit_max = deposits[i]

        if deposits[i] > deposit_max:
            particle_max = particle_id[i]
            deposit_max = deposits[i]

    max_parts_result[sensor_tracking] = particle_max

@njit
def _merge_hits_2(deposit, sensor_idx, particle_idx):
    def process(deposits, sensor_idx):
        # sensor_idx has to be sorted
        sensor_tracking = sensor_idx[0]
        total_sensor_dep = 0.0

        sensor_idx_result = np.zeros_like(sensor_idx) - 1
        deposit_result = np.zeros_like(deposits) - 1
        last = 0

        for i in range(len(deposits)):
            if sensor_idx[i] != sensor_tracking:
                # Accumulate data
                sensor_idx_result[last] = sensor_tracking
                deposit_result[last] = total_sensor_dep
                last += 1

                # Reset state
                sensor_tracking = sensor_idx[i]
                total_sensor_dep = 0.0

            total_sensor_dep += deposits[i]

        sensor_idx_result[last] = sensor_tracking
        deposit_result[last] = total_sensor_dep

        return deposit_result, sensor_idx_result


    sort_idx = np.argsort(particle_idx)
    particle_idx_s = particle_idx[sort_idx]
    deposit_s = deposit[sort_idx]
    sensor_idx_s = sensor_idx[sort_idx]

    sort_idx = np.argsort(sensor_idx)
    particle_idx_s = particle_idx_s[sort_idx]
    deposit_s = deposit_s[sort_idx]
    sensor_idx_s = sensor_idx_s[sort_idx]

    deposit_r, sensor_idx_r = process(particle_idx_s, deposit_s, sensor_idx_s)
    filt = deposit_r != -1
    return deposit_r[filt], sensor_idx_r[filt]

@njit
def _merge_hits(deposit, sensor_idx):
    def process(deposits, sensor_idx):
        # sensor_idx has to be sorted
        sensor_tracking = sensor_idx[0]
        total_sensor_dep = 0.0

        sensor_idx_result = np.zeros_like(sensor_idx) - 1
        deposit_result = np.zeros_like(deposits) - 1
        last = 0

        for i in range(len(deposits)):
            if sensor_idx[i] != sensor_tracking:
                # Accumulate data
                sensor_idx_result[last] = sensor_tracking
                deposit_result[last] = total_sensor_dep
                last += 1

                # Reset state
                sensor_tracking = sensor_idx[i]
                total_sensor_dep = 0.0

            total_sensor_dep += deposits[i]

        sensor_idx_result[last] = sensor_tracking
        deposit_result[last] = total_sensor_dep

        return deposit_result, sensor_idx_result

    sort_idx = np.argsort(sensor_idx)

    deposit_s = deposit[sort_idx]
    sensor_idx_s = sensor_idx[sort_idx]

    deposit_r, sensor_idx_r = process(deposit_s, sensor_idx_s)
    filt = deposit_r != -1
    return deposit_r[filt], sensor_idx_r[filt]

class EventGenerator():
    """
    Generates events from different simulations
    """
    def __init__(self, full_sensor_data, noise_fluctuations=None, cut=0, num_hits_cut=0, reduce=False, area_normed_cut=True,
                 merge_closeby_particles=True, merging_dist_factor=1.5):
        self.full_sensor_data = full_sensor_data
        self.noise_fluctuations = noise_fluctuations
        self.cut = cut
        self.num_hits_cut = num_hits_cut
        self.reduce = reduce
        self.area_normed_cut = area_normed_cut
        self.merge_closeby_particles=merge_closeby_particles
        self.merging_dist_factor = merging_dist_factor
        self.reset()

    def reset(self):
        self.all_sensors_energy = self.full_sensor_data['sensors_x'] * 0.0
        self.all_sensors_x = self.full_sensor_data['sensors_x'] * 1.0
        self.all_sensors_y = self.full_sensor_data['sensors_y'] * 1.0
        self.all_sensors_z = self.full_sensor_data['sensors_z'] * 1.0

        self.all_sensors_eta, self.all_sensors_phi, self.all_sensors_theta = dm.x_y_z_to_eta_phi_theta(self.all_sensors_x, self.all_sensors_y, self.all_sensors_z)
        self.all_sensors_deta = self.full_sensor_data['sensors_deta'] * 1.0
        self.all_sensors_dphi = self.full_sensor_data['sensors_dphi'] * 1.0

        self.all_sensors_start_eta = self.all_sensors_eta - self.all_sensors_deta / 2.0
        self.all_sensors_end_eta = self.all_sensors_eta + self.all_sensors_deta / 2.0

        self.all_sensors_dtheta = dm.angle_diff(dm.eta_to_theta(self.all_sensors_start_eta),dm.eta_to_theta(self.all_sensors_end_eta))

        # self.all_sensors_start_phi = self.all_sensors_phi - self.all_sensors_dphi / 2.0
        # self.all_sensors_end_phi = self.all_sensors_phi + self.all_sensors_dphi / 2.0

        self.all_sensors_area = self.full_sensor_data['sensors_area'] * 1.0

        self.all_sensors_energy = self.all_sensors_x * 0.0
        self.all_sensors_scale = self.full_sensor_data['sensors_scaling'] * 1.0
        # print(np.max(self.all_sensors_area), np.min(self.all_sensors_area))
        # 0/0
        self.all_sensors_area_norm = self.all_sensors_area / np.max(self.all_sensors_area)

        self.truth_assignment = np.zeros_like(self.all_sensors_x, np.int32) - 1

        self.rechit_energy  = self.all_sensors_x * 0.0

        if self.noise_fluctuations is not None:
            typ = self.noise_fluctuations[0]
            if typ == 'type_a':
                m = self.noise_fluctuations[1]
                s = self.noise_fluctuations[2]

                rechit_energy_fluctuation = np.maximum(0, np.random.normal(m, s,
                                                                           size=self.rechit_energy.shape))  # / rechit_thickness_norm
                self.rechit_energy += rechit_energy_fluctuation * 1.0

                noisy_hits = np.sum(self.rechit_energy / (self.all_sensors_area_norm if self.area_normed_cut else 1.0) > self.cut)
                print("Num noise", noisy_hits)

        self.truth_rechit_deposit_max = self.rechit_energy * 1.0
        self.truth_assignment_vertex_position_x = self.all_sensors_x * 0.0
        self.truth_assignment_vertex_position_y = self.all_sensors_x * 0.0
        self.truth_assignment_vertex_position_z = self.all_sensors_x * 0.0
        self.truth_assignment_momentum_direction_x = self.all_sensors_x * 0.0
        self.truth_assignment_momentum_direction_y = self.all_sensors_x * 0.0
        self.truth_assignment_momentum_direction_z = self.all_sensors_x * 0.0
        self.truth_assignment_energy = self.rechit_energy.copy()
        self.truth_assignment_pdgid = np.zeros_like(self.all_sensors_x, np.int32)
        self.truth_assignment_energy_dep = self.rechit_energy.copy()
        self.truth_assignment_energy_dep_all = self.all_sensors_x * 0.0
        self.overall_particle_id = 0
        self.truth_assignment_x = self.rechit_energy * 0.0
        self.truth_assignment_y = self.rechit_energy * 0.0
        self.truth_assignment_z = self.rechit_energy * 0.0

        self.merging_occurred = False
        self.merging_occured_places = self.rechit_energy * 0.0

        self.simulations=[]


    def _gather_rechit_energy(self, simulations, from_particle_hit_data=False):
        if from_particle_hit_data:
            """
            If some rechit doesn't have a particle assigned to it at time of geant simulation, that might be an issue.
            So far its never the case. So it should be good
            """
            rechit_energy = tf.convert_to_tensor(self.rechit_energy * 1.0)
            for simulation in simulations:
                rechit_energy = tf.tensor_scatter_nd_add(rechit_energy, simulation['hit_particle_sensor_idx'][:, tf.newaxis], simulation['hit_particle_deposit'])

            self.rechit_energy = rechit_energy.numpy()
        else:
            for simulation in simulations:
                self.rechit_energy[simulation['rechit_idx'].astype(np.int32)] = self.rechit_energy[
                                                                               simulation['rechit_idx'].astype(np.int32)] + \
                                                                           simulation['rechit_energy']

    def _filter_simulations(self, simulations, phase_cut=None, eta_cut=None):
        if len(simulations) == 1 and (phase_cut==None and eta_cut==None):
            return simulations[0]

        if eta_cut is not None:
            raise NotImplementedError('NOT IMPLEMENTED')

        reduced_simulation = {}
        reduced_simulation['hit_particle_id'] = []
        reduced_simulation['hit_particle_deposit'] = []
        reduced_simulation['hit_particle_sensor_idx'] = []
        reduced_simulation['particles_vertex_position_x'] = []
        reduced_simulation['particles_vertex_position_y'] = []
        reduced_simulation['particles_vertex_position_z'] = []
        reduced_simulation['particles_pdgid'] = []
        reduced_simulation['particles_momentum_direction_x'] = []
        reduced_simulation['particles_momentum_direction_y'] = []
        reduced_simulation['particles_momentum_direction_z'] = []
        reduced_simulation['particles_kinetic_energy'] = []
        reduced_simulation['particles_total_energy_deposited_all'] = []
        reduced_simulation['particles_only_minbias'] = []
        reduced_simulation['particles_shower_class'] = []


        reduced_simulation['particles_first_active_impact_position_x'] = []
        reduced_simulation['particles_first_active_impact_position_y'] = []
        reduced_simulation['particles_first_active_impact_position_z'] = []
        reduced_simulation['particles_first_active_impact_sensor_idx'] = []

        reduced_simulation['particles_first_active_impact_momentum_direction_x'] = []
        reduced_simulation['particles_first_active_impact_momentum_direction_y'] = []
        reduced_simulation['particles_first_active_impact_momentum_direction_z'] = []

        if phase_cut != None:
            keep_phi_start = np.random.uniform(0, 2 * np.pi)
            keep_phi_end = np.fmod(keep_phi_start + phase_cut, 2 * np.pi)

        index = 0
        for simulation in simulations:
            num_particles_available = len(simulation['particles_vertex_position_x'])

            if num_particles_available == 0:
                continue

            hits_particle_id = simulation['hit_particle_id']
            hit_particle_deposit = simulation['hit_particle_deposit']
            hit_particle_sensor_idx = simulation['hit_particle_sensor_idx']

            particles_first_active_impact_position_eta,particles_first_active_impact_position_phi,_ = \
                dm.x_y_z_to_eta_phi_theta(np.array(simulation['particles_first_active_impact_position_x']),
                                      np.array(simulation['particles_first_active_impact_position_y']),
                                      np.array(simulation['particles_first_active_impact_position_z']))


            keep_this = np.ones_like(np.array(simulation['particles_first_active_impact_position_x']), np.bool)
            if phase_cut != None:
                # keep_phi_start = 0.4*np.pi/2 # TODO: np.random.uniform()
                particles_first_active_impact_position_phi = np.where(particles_first_active_impact_position_phi<0, particles_first_active_impact_position_phi+2*np.pi, particles_first_active_impact_position_phi)
                if keep_phi_start > keep_phi_end:
                    keep_this = np.logical_or(particles_first_active_impact_position_phi < keep_phi_end, particles_first_active_impact_position_phi > keep_phi_start)
                else:
                    keep_this = np.logical_and(particles_first_active_impact_position_phi < keep_phi_end,
                                              particles_first_active_impact_position_phi > keep_phi_start)

            # print("Z", np.mean(keep_this))

                # for i in range(len(keep_this)):
                #     x = particles_first_active_impact_position_phi[i]
                #     v = lambda x :np.rad2deg(x)
                #     print("Z", v(x),v(keep_phi_start), v(keep_phi_end), x < keep_phi_end, x> keep_phi_start)


            for i in range(num_particles_available):
                if not keep_this[i]:
                    continue

                condition = hits_particle_id==i
                num = np.sum(condition)
                if num == 0:
                    continue

                reduced_simulation['hit_particle_id'] += [np.ones(num) * index]
                index += 1
                reduced_simulation['hit_particle_deposit'] += [hit_particle_deposit[condition]]
                reduced_simulation['hit_particle_sensor_idx'] += [hit_particle_sensor_idx[condition]]

                reduced_simulation['particles_vertex_position_x'] += [simulation['particles_vertex_position_x'][i]]
                reduced_simulation['particles_vertex_position_y'] += [simulation['particles_vertex_position_y'][i]]
                reduced_simulation['particles_vertex_position_z'] += [simulation['particles_vertex_position_z'][i]]
                reduced_simulation['particles_pdgid'] += [simulation['particles_pdgid'][i]]
                reduced_simulation['particles_momentum_direction_x'] += [simulation['particles_momentum_direction_x'][i]]
                reduced_simulation['particles_momentum_direction_y'] += [simulation['particles_momentum_direction_y'][i]]
                reduced_simulation['particles_momentum_direction_z'] += [simulation['particles_momentum_direction_z'][i]]
                reduced_simulation['particles_kinetic_energy'] += [simulation['particles_kinetic_energy'][i]]
                reduced_simulation['particles_total_energy_deposited_all'] += [simulation['particles_total_energy_deposited_all'][i]]

                reduced_simulation['particles_first_active_impact_position_x'] += [simulation['particles_first_active_impact_position_x'][i]]
                reduced_simulation['particles_first_active_impact_position_y'] += [simulation['particles_first_active_impact_position_y'][i]]
                reduced_simulation['particles_first_active_impact_position_z'] += [simulation['particles_first_active_impact_position_z'][i]]
                reduced_simulation['particles_first_active_impact_sensor_idx'] += [simulation['particles_first_active_impact_sensor_idx'][i]]
                reduced_simulation['particles_only_minbias'] += [simulation['particles_only_minbias'][i]]
                reduced_simulation['particles_shower_class'] += [simulation['particles_shower_class'][i]]

                reduced_simulation['particles_first_active_impact_momentum_direction_x'] += [simulation['particles_first_active_impact_momentum_direction_x'][i]]
                reduced_simulation['particles_first_active_impact_momentum_direction_y'] += [simulation['particles_first_active_impact_momentum_direction_y'][i]]
                reduced_simulation['particles_first_active_impact_momentum_direction_z'] += [simulation['particles_first_active_impact_momentum_direction_z'][i]]

        if len(reduced_simulation['hit_particle_id']) > 0:
            reduced_simulation['hit_particle_id'] = np.concatenate(reduced_simulation['hit_particle_id'], axis=0)
            reduced_simulation['hit_particle_deposit'] = np.concatenate(reduced_simulation['hit_particle_deposit'], axis=0)
            reduced_simulation['hit_particle_sensor_idx'] = np.concatenate(reduced_simulation['hit_particle_sensor_idx'], axis=0)
        else:
            reduced_simulation['hit_particle_id'] = np.array([], np.int32)
            reduced_simulation['hit_particle_deposit'] = np.array([], np.float)
            reduced_simulation['hit_particle_sensor_idx'] = np.array([], np.float)

        return reduced_simulation

    def _merge_particles(self, simulation):
        t1 = time.time()
        use_tf=True

        first_sensor_indices = simulation['particles_first_active_impact_sensor_idx']
        ex2 = np.array(simulation['particles_first_active_impact_position_x'])
        ey2 = np.array(simulation['particles_first_active_impact_position_y'])
        ez2 = np.array(simulation['particles_first_active_impact_position_z'])

        ex = ex2
        ey = ey2
        ez = ez2

        # Convert them to eta,phi
        eeta, ephi, _ = dm.x_y_z_to_eta_phi_theta(ex, ey, ez)

        if use_tf:
            eeta, ephi = tf.convert_to_tensor(eeta), tf.convert_to_tensor(ephi)

        # sensors_eta, sensors_phi, _ = dm.x_y_z_to_eta_phi_theta(sensor_locations[:, 0],sensor_locations[:, 1],sensor_locations[:, 2])

        if use_tf:
            phi_distance = dm.angle_diff_tf(ephi[:, tf.newaxis], ephi[tf.newaxis, :])
        else:
            phi_distance = dm.angle_diff(ephi[:, np.newaxis], ephi[np.newaxis, :])

        if use_tf:
            eta_distance = tf.abs(eeta[:, np.newaxis] - eeta[np.newaxis, :])
        else:
            eta_distance = np.abs(eeta[:, np.newaxis] - eeta[np.newaxis, :])

        # theta_distance = dm.angle_diff(dm.eta_to_theta(eeta[:, np.newaxis]), dm.eta_to_theta(eeta[np.newaxis, :]))
        # distance_matrix = np.sqrt(angle_distance**2 + eta_distance**2)

        if use_tf:
            picked_sensors_dphi = tf.maximum(self.all_sensors_dphi[first_sensor_indices][:, np.newaxis],
                                             self.all_sensors_dphi[first_sensor_indices][np.newaxis, :])
        else:
            picked_sensors_dphi = np.maximum(self.all_sensors_dphi[first_sensor_indices][:, np.newaxis], self.all_sensors_dphi[first_sensor_indices][np.newaxis, :])

        close_in_phi =  phi_distance < picked_sensors_dphi * self.merging_dist_factor

        # picked_sensors_dtheta = np.maximum(self.all_sensors_dtheta[first_sensor_indices][:, np.newaxis], self.all_sensors_dtheta[first_sensor_indices][np.newaxis, :])
        # close_in_theta =  theta_distance < picked_sensors_dtheta * self.merging_dist_factor
        if use_tf:
            picked_sensors_deta = tf.maximum(self.all_sensors_deta[first_sensor_indices][:, np.newaxis], self.all_sensors_deta[first_sensor_indices][np.newaxis, :])
        else:
            picked_sensors_deta = np.maximum(self.all_sensors_deta[first_sensor_indices][:, np.newaxis], self.all_sensors_deta[first_sensor_indices][np.newaxis, :])

        close_in_eta =  eta_distance < picked_sensors_deta * self.merging_dist_factor

        mx = np.array(simulation['particles_first_active_impact_momentum_direction_x'])
        my = np.array(simulation['particles_first_active_impact_momentum_direction_y'])
        mz = np.array(simulation['particles_first_active_impact_momentum_direction_z'])

        momentum_vectors =tf.concat((mx[..., tf.newaxis],my[..., tf.newaxis],mz[..., tf.newaxis]), axis=-1)

        angle_distances = dm.nxnAngles_tf(momentum_vectors, momentum_vectors)
        limit = np.deg2rad(5.0)
        close_in_angle = angle_distances < limit

        # angle_projection += [dm.angle_between_vectors([ex, ey, ez], [mx, my, mz])]

        connection_adjacency_matrix = tf.logical_and(close_in_phi, close_in_eta)
        connection_adjacency_matrix = tf.logical_and(close_in_angle, connection_adjacency_matrix)

        if use_tf:
            connection_adjacency_matrix = connection_adjacency_matrix.numpy()

        num_showers_after_merging, labels = connected_components(connection_adjacency_matrix, directed=False)

        # print("Num merged ", np.max(labels), len(connection_adjacency_matrix))

        study_variable = []
        print("\t\tMerging part 1 took", time.time()-t1,"seconds")
        t1 = time.time()
        if num_showers_after_merging != len(labels):
            reduced_simulation = {}
            reduced_simulation['hit_particle_id'] = []
            reduced_simulation['hit_particle_deposit'] = []
            reduced_simulation['hit_particle_sensor_idx'] = []
            reduced_simulation['particles_vertex_position_x'] = []
            reduced_simulation['particles_vertex_position_y'] = []
            reduced_simulation['particles_vertex_position_z'] = []
            reduced_simulation['particles_pdgid'] = []
            reduced_simulation['particles_momentum_direction_x'] = []
            reduced_simulation['particles_momentum_direction_y'] = []
            reduced_simulation['particles_momentum_direction_z'] = []
            reduced_simulation['particles_kinetic_energy'] = []
            reduced_simulation['particles_total_energy_deposited_all'] = []
            reduced_simulation['particles_only_minbias'] = []
            reduced_simulation['merged'] = []
            reduced_simulation['particles_shower_class'] = []

            unique_labels = np.unique(labels)

            hits_particle_id = simulation['hit_particle_id'].astype(np.int64)
            hits_particle_deposit = simulation['hit_particle_deposit']
            hits_particle_sensor_idx = simulation['hit_particle_sensor_idx']

            if use_tf:
                hits_particle_id = tf.convert_to_tensor(hits_particle_id)
                hits_particle_deposit = tf.convert_to_tensor(hits_particle_deposit)
                hits_particle_sensor_idx = tf.convert_to_tensor(hits_particle_sensor_idx)

            hits_particle_id = tf.gather_nd(labels, hits_particle_id[..., tf.newaxis])
            hits_particle_id, hits_particle_sensor_idx, hits_particle_deposit = merge_hits(hits_particle_id.numpy(), hits_particle_sensor_idx.numpy(), hits_particle_deposit.numpy())

            for u in unique_labels:
                combine_these = np.argwhere(labels == u)[:, 0]
                bigger = combine_these[0] # TODO: FIX THIS? combine_these[np.argmax([np.sum(hits_particle_deposit[hits_particle_id==p]) for p in combine_these])]


                if len(combine_these) > 1:
                    # print(combine_these)
                    reduced_simulation['merged'].append(True)
                    # study_variable_ = []
                    # for c_i in range(len(combine_these)):
                    #     for d_i in range(len(combine_these)):
                    #         if d_i > c_i:
                    #             x = float(angle_distances[combine_these[c_i], combine_these[d_i]].numpy())
                    #             study_variable_.append(x)
                    # study_variable += study_variable_
                    # print(len(combine_these), np.max(np.rad2deg(study_variable_)), np.rad2deg(study_variable_))
                else:
                    reduced_simulation['merged'].append(False)

                reduced_simulation['particles_vertex_position_x'] += [simulation['particles_vertex_position_x'][bigger]]
                reduced_simulation['particles_vertex_position_y'] += [simulation['particles_vertex_position_y'][bigger]]
                reduced_simulation['particles_vertex_position_z'] += [simulation['particles_vertex_position_z'][bigger]]
                reduced_simulation['particles_pdgid'] += [simulation['particles_pdgid'][bigger]]
                reduced_simulation['particles_momentum_direction_x'] += [simulation['particles_momentum_direction_x'][bigger]]
                reduced_simulation['particles_momentum_direction_y'] += [simulation['particles_momentum_direction_y'][bigger]]
                reduced_simulation['particles_momentum_direction_z'] += [simulation['particles_momentum_direction_z'][bigger]]
                reduced_simulation['particles_only_minbias'] += [np.all([simulation['particles_only_minbias'][x] for x in combine_these])]
                reduced_simulation['particles_kinetic_energy'] += [sum([simulation['particles_kinetic_energy'][x] for x in combine_these])]
                reduced_simulation['particles_total_energy_deposited_all'] += [sum([simulation['particles_total_energy_deposited_all'][x] for x in combine_these])]

                eneries = np.array([simulation['particles_kinetic_energy'][x] for x in combine_these])
                is_em = np.array([simulation['particles_pdgid'][x] in {22, 11, -11} for x in combine_these])
                is_pion_plus = np.array([simulation['particles_pdgid'][x]==211 for x in combine_these])
                is_mix = np.array([simulation['particles_pdgid'][x] in {22, 11, -11, 211}  for x in combine_these])
                # is_unclassified = np.array([x for x in simulation['particles_pdgid']])

                assigned_class = -1
                if np.sum(eneries*is_em) > 0.95*np.sum(eneries):
                    assigned_class = 0
                elif np.sum(eneries*is_pion_plus) > 0.95*np.sum(eneries):
                    assigned_class = 1
                elif np.sum(eneries*is_mix) > 0.95*np.sum(eneries):
                    assigned_class = 2

                # print("Assigned class", assigned_class, [simulation['particles_shower_class'][x] for x in combine_these],[simulation['particles_pdgid'][x] for x in combine_these],[simulation['particles_kinetic_energy'][x] for x in combine_these], is_em)

                reduced_simulation['particles_shower_class'] += [assigned_class]



            self.merging_occurred = True
            reduced_simulation['hit_particle_id'] = hits_particle_id
            reduced_simulation['hit_particle_deposit'] = hits_particle_deposit
            reduced_simulation['hit_particle_sensor_idx'] = hits_particle_sensor_idx

            # print(study_variable)
            # study_variable = np.rad2deg(study_variable)
            # plt.hist(study_variable)
            # plt.show()

            print("\t\tMerging part 2 took", time.time() - t1, "seconds")
            return reduced_simulation
        else:
            simulation['merged'] = [False] * len(simulation['particles_vertex_position_x'])
            print("\t\tMerging part 2 took", time.time() - t1, "seconds")
            return simulation

    def get_particle_that_max_dep_on_sensor(self,particle_id, sensor_idx, deposits):
        sort_idx = np.argsort(sensor_idx)
        sensor_idx = sensor_idx[sort_idx]
        particle_id = particle_id[sort_idx]
        deposits = deposits[sort_idx]

        sensors_max_result = (self.rechit_energy * 0.0).astype(np.int32) - 1

        _get_particle_that_max_dep_on_sensor(particle_id, sensor_idx, deposits, sensors_max_result)

        return sensors_max_result

    def _gather_rechits(self, simulation):
        num_particles_available = len(simulation['particles_vertex_position_x'])

        hits_particle_id = simulation['hit_particle_id'].astype(np.int32)
        hit_particle_deposit = simulation['hit_particle_deposit']
        hit_particle_sensor_idx = simulation['hit_particle_sensor_idx'].astype(np.int32)

        hit_particle_deposit = hit_particle_deposit * self.all_sensors_scale[hit_particle_sensor_idx]

        sort_idx = tf.argsort(hits_particle_id)
        hits_particle_id = tf.gather_nd(hits_particle_id, sort_idx[..., tf.newaxis])
        hit_particle_deposit = tf.gather_nd(hit_particle_deposit, sort_idx[..., tf.newaxis])
        hit_particle_sensor_idx = tf.gather_nd(hit_particle_sensor_idx, sort_idx[..., tf.newaxis])
        segment_ids = hits_particle_id*1

        indexing = hit_particle_sensor_idx[..., tf.newaxis]

        # hit_particle_deposit = tf.RaggedTensor.from_value_rowids(hit_particle_deposit, hits_particle_id)
        # hit_particle_sensor_idx = tf.RaggedTensor.from_value_rowids(hit_particle_sensor_idx, hits_particle_id)
        # hits_particle_id = tf.RaggedTensor.from_value_rowids(hits_particle_id, hits_particle_id)

        deposit_sum_particles = tf.math.segment_sum(hit_particle_deposit, segment_ids)
        particles_true_x = tf.math.segment_sum(tf.gather_nd(self.all_sensors_x, indexing) * hit_particle_deposit, segment_ids) / deposit_sum_particles
        particles_true_y = tf.math.segment_sum(tf.gather_nd(self.all_sensors_y, indexing) * hit_particle_deposit, segment_ids) / deposit_sum_particles
        particles_true_z = tf.math.segment_sum(tf.gather_nd(self.all_sensors_z, indexing) * hit_particle_deposit, segment_ids) / deposit_sum_particles

        x = np.concatenate(((self.rechit_energy*0).astype(np.int32)-1, hits_particle_id.numpy()), axis=0)
        y = np.concatenate((np.arange(len(self.rechit_energy)), hit_particle_sensor_idx.numpy()), axis=0)
        z = np.concatenate((self.rechit_energy, hit_particle_deposit.numpy()), axis=0)
        max_part_dep = self.get_particle_that_max_dep_on_sensor(x, y, z)

        # Add -1 at the end -1 indexing will take you to the last element which is what is for the noise
        particles_vertex_position_x = np.concatenate((np.array(simulation['particles_vertex_position_x']), [0]), axis=-1)
        particles_vertex_position_y = np.concatenate((np.array(simulation['particles_vertex_position_y']), [0]), axis=-1)
        particles_vertex_position_z = np.concatenate((np.array(simulation['particles_vertex_position_z']), [0]), axis=-1)
        particles_momentum_direction_x = np.concatenate((np.array(simulation['particles_momentum_direction_x']), [0]), axis=-1)
        particles_momentum_direction_y = np.concatenate((np.array(simulation['particles_momentum_direction_y']), [0]), axis=-1)
        particles_momentum_direction_z = np.concatenate((np.array(simulation['particles_momentum_direction_z']), [0]), axis=-1)
        if self.merge_closeby_particles:
            merged = np.concatenate((np.array(simulation['merged']), [False]), axis=-1)
        particles_pdgid = np.concatenate((np.array(simulation['particles_pdgid']), [0]), axis=-1)
        particles_kinetic_energy = np.concatenate((np.array(simulation['particles_kinetic_energy']), [0]), axis=-1)
        particles_total_energy_deposited_all = np.concatenate((np.array(simulation['particles_total_energy_deposited_all']), [0]), axis=-1)

        particles_only_minbias = np.concatenate((np.array(simulation['particles_only_minbias']), [False]), axis=-1)
        particles_shower_class = np.concatenate((np.array(simulation['particles_shower_class']), [-1]), axis=-1)

        particles_true_x = np.concatenate((particles_true_x.numpy(), [0]), axis=-1)
        particles_true_y = np.concatenate((particles_true_y.numpy(), [0]), axis=-1)
        particles_true_z = np.concatenate((particles_true_z.numpy(), [0]), axis=-1)
        deposit_sum_particles = np.concatenate((deposit_sum_particles.numpy(), [0]), axis=-1)

        self.truth_assignment = max_part_dep * 1
        self.truth_assignment_vertex_position_x = particles_vertex_position_x[max_part_dep]
        self.truth_assignment_vertex_position_y = particles_vertex_position_y[max_part_dep]
        self.truth_assignment_vertex_position_z = particles_vertex_position_z[max_part_dep]
        self.truth_assignment_x = particles_true_x[max_part_dep]
        self.truth_assignment_y = particles_true_y[max_part_dep]
        self.truth_assignment_z = particles_true_z[max_part_dep]
        self.truth_assignment_momentum_direction_x = particles_momentum_direction_x[max_part_dep]
        self.truth_assignment_momentum_direction_y = particles_momentum_direction_y[max_part_dep]
        self.truth_assignment_momentum_direction_z = particles_momentum_direction_z[max_part_dep]
        if self.merge_closeby_particles:
            self.merging_occured_places =  merged[max_part_dep]
        self.truth_assignment_pdgid = particles_pdgid[max_part_dep]
        self.truth_assignment_energy = particles_kinetic_energy[max_part_dep]
        self.truth_assignment_energy_dep = deposit_sum_particles[max_part_dep]
        self.truth_assignment_energy_dep_all = particles_total_energy_deposited_all[max_part_dep]
        self.truth_assignment_only_minbias = particles_only_minbias[max_part_dep]
        self.truth_assignment_shower_class = particles_shower_class[max_part_dep]


    def _gather_event_data(self):
        if self.cut > 0:
            energy_by_area = self.rechit_energy / (self.all_sensors_area_norm if self.area_normed_cut else 1.0)
            cut_off = energy_by_area < self.cut
            self.rechit_energy[cut_off] = 0.0
            self.truth_assignment[self.rechit_energy == 0.] = -1

        self.rechit_energy = self.rechit_energy * self.all_sensors_scale

        uniques, counts = np.unique(self.truth_assignment, return_counts=True)
        counts = counts[uniques != -1]
        uniques = uniques[uniques != -1]

        if self.num_hits_cut > 1:
            for u, c in zip(uniques, counts):
                if u == -1:
                    continue
                if c <= self.num_hits_cut:
                    self.truth_assignment[self.truth_assignment == u] = -1

        # truth_assignment_energy_dep = truth_assignment_energy_dep * 0.
        truth_assignment_energy_dep_end = self.truth_assignment_energy_dep * 0.
        for u in uniques:
            truth_assignment_energy_dep_end[self.truth_assignment==u] = np.sum(self.rechit_energy[self.truth_assignment==u])

        truth_data = [self.truth_rechit_deposit_max,
                      self.truth_assignment_vertex_position_x,
                      self.truth_assignment_vertex_position_y,
                      self.truth_assignment_vertex_position_z,
                      self.truth_assignment_x,
                      self.truth_assignment_y,
                      self.truth_assignment_z,
                      self.truth_assignment_momentum_direction_x,
                      self.truth_assignment_momentum_direction_y,
                      self.truth_assignment_momentum_direction_z,
                      self.truth_assignment_pdgid,
                      self.truth_assignment_energy,
                      self.truth_assignment_energy_dep,
                      truth_assignment_energy_dep_end,
                      self.truth_assignment_energy_dep_all, ]

        for t in truth_data:
            t[self.truth_assignment == -1] = 0

        result = {
            'rechit_x': self.all_sensors_x / 10.,
            'rechit_y': self.all_sensors_y / 10.,
            'rechit_z': self.all_sensors_z / 10.,
            'rechit_sensor_idx': np.arange(len(self.all_sensors_x)),
            'rechit_layer': self.full_sensor_data['sensors_active_layer_num'] * 1.0,
            'rechit_area': self.all_sensors_area / 100.,
            'rechit_energy': self.rechit_energy,
            'truth_assignment': self.truth_assignment,
            'truth_assignment_hit_dep': self.truth_rechit_deposit_max,
            'truth_assignment_vertex_position_x': self.truth_assignment_vertex_position_x / 10.,
            'truth_assignment_vertex_position_y': self.truth_assignment_vertex_position_y / 10.,
            'truth_assignment_vertex_position_z': self.truth_assignment_vertex_position_z / 10.,
            'truth_assignment_x': self.truth_assignment_x / 10.,
            'truth_assignment_y': self.truth_assignment_y / 10.,
            'truth_assignment_z': self.truth_assignment_z / 10.,
            'truth_assignment_momentum_direction_x': self.truth_assignment_momentum_direction_x / 10.,
            'truth_assignment_momentum_direction_y': self.truth_assignment_momentum_direction_y / 10.,
            'truth_assignment_momentum_direction_z': self.truth_assignment_momentum_direction_z / 10.,
            'truth_assignment_pdgid': self.truth_assignment_pdgid,
            'truth_assignment_energy': self.truth_assignment_energy,
            'truth_assignment_energy_dep': self.truth_assignment_energy_dep,
            'truth_assignment_energy_dep_end': truth_assignment_energy_dep_end,
            'truth_assignment_energy_dep_all': self.truth_assignment_energy_dep_all,
            'truth_assignment_shower_class': self.truth_assignment_shower_class,
            'truth_assignment_only_minbias': self.truth_assignment_only_minbias,
        }
        if self.merge_closeby_particles:
            result['truth_merging_occurred'] = self.merging_occured_places

        if self.reduce:
            filt = self.rechit_energy > 0
            result = {k: v[filt] for k, v in result.items()}

        return result

    def _process_muons(self, simulation):
        simulation['particles_kinetic_energy'] = [simulation['particles_kinetic_energy'][x] if simulation['particles_total_energy_deposited_all'][x] != 13 else simulation['particles_pdgid'][x] for x in np.arange(len(simulation['particles_kinetic_energy']))]
        return simulation


    def _draw_experiments(self, simulation):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        print("VVV", np.max(simulation['particles_first_active_impact_position_z']))

        angle_projection = []

        # study = {424,124,494}
        # study = {124}
        # study_l = [424, 124, 494]
        # study_l = [424]
        study_l = [424, 124, 494]
        c_l = ['red', 'blue', 'green']


        hit_particle_id = simulation['hit_particle_id']
        hit_particle_sensor_idx = simulation['hit_particle_sensor_idx']

        for s,c in zip(study_l, c_l):
            filt = hit_particle_id==s
            hit_particle_sensor_idx_2 = hit_particle_sensor_idx[filt].astype(np.int32)
            hit_particle_x = self.all_sensors_x[hit_particle_sensor_idx_2]
            hit_particle_y = self.all_sensors_y[hit_particle_sensor_idx_2]
            hit_particle_z = self.all_sensors_z[hit_particle_sensor_idx_2]
            hit_particle_e = self.all_sensors_z[hit_particle_sensor_idx_2]
            hit_particle_c = self.all_sensors_z[hit_particle_sensor_idx]
            ax.scatter(hit_particle_z, hit_particle_x, hit_particle_y, s=0.3, c=c)
            # plt.show()


        for i in range(len(simulation['particles_kinetic_energy'])):
            pos_a = simulation['particles_kinetic_energy']
            mom = None

            if not i in study_l:
                continue

            # print(simulation.keys())

            ex = np.array(simulation['particles_first_active_impact_position_x'][i])
            ey = np.array(simulation['particles_first_active_impact_position_y'][i])
            ez = np.array(simulation['particles_first_active_impact_position_z'][i])

            emx = np.array(simulation['particles_first_active_impact_momentum_direction_x'][i])
            emy = np.array(simulation['particles_first_active_impact_momentum_direction_y'][i])
            emz = np.array(simulation['particles_first_active_impact_momentum_direction_z'][i])

            angle_projection += [dm.angle_between_vectors([ex,ey,ez],[emx,emy,emz])]

            s = np.sqrt(emx ** 2 + emy ** 2 + emz ** 2)

            ex2 = ex + 100 * emx / s
            ey2 = ey + 100 * emy / s
            ez2 = ez + 100 * emz / s

            print(emx,emy,emz, np.sqrt(emx**2 + emy**2 + emz**2))
            print(ex,ey,ez)
            print(ex2,ey2,ez2,'\n')

            # print()
            # v = np.where(np.array(study_l) == i)[0][0]
            # c = c_l[v]
            ax.plot([ez, ez2], [ex, ex2], [ey,ey2], c=c)


            # ax.plot([ez, ez2], [ex, ex2], [ey,ey2])

        ax.set_xlabel('z (mm)')
        ax.set_ylabel('x (mm)')
        ax.set_zlabel('y (mm)')
        # ax.scatter(z, x, y, s=np.log(e+1)*100)
        # ax.plot([sz, ez], [sx, ex], [sy, ey])
        plt.show()

        angle_projection = np.rad2deg(angle_projection)
        print("MIN MAX ANGLE", np.min(angle_projection), np.max(angle_projection))
        plt.hist(angle_projection[angle_projection<5], bins=20)
        plt.show()

    def _attach_minbias_data(self, simulations, minbias=False):
        for s in simulations:
            s['particles_only_minbias'] = np.array([minbias for _ in range(len(s['particles_vertex_position_x']))])
        return simulations

    def _attach_shower_class_data(self, simulations):
        for simulation in simulations:
            pdgid = simulation['particles_pdgid'].tolist()
            shower_class = [-1 for x in simulation['particles_pdgid']]
            shower_class = [0 if pdg in {22, 11, -11} else c for (pdg,c) in zip(pdgid, shower_class)]
            shower_class = [1 if pdg == 211 else c for (pdg,c) in zip(pdgid, shower_class)]
            simulation['particles_shower_class'] = shower_class
        return simulations

    def add(self, simulations, phase_cut=None, eta_cut=None, minbias=False):
        simulations = self._attach_minbias_data(simulations, minbias)
        simulations = self._attach_shower_class_data(simulations)

        still_to_gather=True
        if phase_cut==None and eta_cut==None:
            t1 = time.time()
            self._gather_rechit_energy(simulations)
            print("\tGathering rechit energy took", time.time()-t1,"seconds")
            still_to_gather=False

        t1 = time.time()
        reduced_simulation = self._filter_simulations(simulations, phase_cut=phase_cut, eta_cut=eta_cut)
        print("\tReduction took", time.time()-t1,"seconds")

        if still_to_gather:
            self._gather_rechit_energy([reduced_simulation], from_particle_hit_data=True)

        self.simulations += [reduced_simulation]

    def process(self, reset_after=True):
        reduced_simulation = self._filter_simulations(self.simulations)
        reduced_simulation = self._process_muons(reduced_simulation)

        # self._draw_experiments(reduced_simulation)

        t1 = time.time()
        if self.merge_closeby_particles:
            reduced_simulation = self._merge_particles(reduced_simulation)
        print("\tTruth merging took", time.time()-t1,"seconds")


        t1 = time.time()
        self._gather_rechits(reduced_simulation)
        print("\tAddition took", time.time()-t1,"seconds")

        d = self._gather_event_data()
        if reset_after:
            self.reset()
        return d

    def did_merging_occur(self):
        return self.merging_occurred



# def combine_events(main_particle_simulation, pu_simulations, full_sensor_data, noise_fluctuations=None, cut=0, num_hits_cut=0, reduce=False, area_normed_cut=True):
#     """
#     Combine PU and particles to generate a full event
#
#     :param main_particle_simulation: Main particle simulation result
#     :param pu_simulations: All the PU simulation results
#     :param full_sensor_data: Sensor data sensor locations of all the sensors and not just hits
#     :param noise_fluctuations: A tuple with mean and std and None for no noise fluctuation
#     :return:
#     """
#     rechit_x = full_sensor_data['sensors_x']
#     rechit_y = full_sensor_data['sensors_y']
#     rechit_z = full_sensor_data['sensors_z']
#     rechit_layer = full_sensor_data['sensors_active_layer_num']
#     rechit_area = full_sensor_data['sensors_area']
#     # rechit_area = np.power(rechit_area, 3/2)
#     rechit_energy = rechit_x * 0.0
#     rechit_scale = full_sensor_data['sensors_scaling']
#     rechit_area_norm = rechit_area / np.max(rechit_area)
#
#     rechit_thickness = full_sensor_data['sensors_thickness']
#     rechit_thickness_norm = rechit_thickness / np.max(rechit_thickness)
#
#     truth_assignment = np.zeros_like(rechit_x, np.int32) - 1
#
#
#     if noise_fluctuations is not None:
#         typ = noise_fluctuations[0]
#         if typ == 'type_a':
#             m = noise_fluctuations[1]
#             s = noise_fluctuations[2]
#
#             rechit_energy_fluctuation = np.maximum(0, np.random.normal(m, s, size=rechit_energy.shape)) #/ rechit_thickness_norm
#             rechit_energy = rechit_energy_fluctuation * 1.0
#
#             noisy_hits = np.sum(rechit_energy / (rechit_area_norm if area_normed_cut else 1.0) > cut)
#             print("Num noise", noisy_hits)
#
#     truth_rechit_deposit_max = rechit_energy * 1.0
#     truth_assignment_vertex_position_x = rechit_x * 0.0
#     truth_assignment_vertex_position_y = rechit_x * 0.0
#     truth_assignment_vertex_position_z = rechit_x * 0.0
#     truth_assignment_momentum_direction_x = rechit_x * 0.0
#     truth_assignment_momentum_direction_y = rechit_x * 0.0
#     truth_assignment_momentum_direction_z = rechit_x * 0.0
#     truth_assignment_energy = rechit_energy.copy()
#     truth_assignment_pdgid =np.zeros_like(rechit_x, np.int32)
#     truth_assignment_energy_dep = rechit_energy.copy()
#     truth_assignment_energy_dep_all = rechit_x * 0.0
#     overall_particle_id = 0
#
#
#     truth_assignment_x = rechit_energy * 0.0
#     truth_assignment_y = rechit_energy * 0.0
#     truth_assignment_z = rechit_energy * 0.0
#
#     def add(simulation):
#         nonlocal overall_particle_id
#         num_particles_available = len(simulation['particles_vertex_position_x'])
#
#         hits_particle_id = simulation['hit_particle_id']
#         hit_particle_deposit = simulation['hit_particle_deposit']
#         hit_particle_sensor_idx = simulation['hit_particle_sensor_idx'].astype(np.int32)
#
#         rechit_energy[simulation['rechit_idx'].astype(np.int32)] = rechit_energy[
#                                                                        simulation['rechit_idx'].astype(np.int32)] + \
#                                                                    simulation['rechit_energy']
#         for i in range(num_particles_available):
#             condition = hits_particle_id == i
#             if np.sum(condition) == 0:
#                 continue
#
#             filt_rechits_idx = hit_particle_sensor_idx[condition]
#             filt_rechits_deposit = hit_particle_deposit[condition]
#
#             dep_scaling = rechit_scale[filt_rechits_idx]
#             dep_energy_part = np.sum(filt_rechits_deposit * dep_scaling)
#
#             true_x = np.sum(rechit_x[filt_rechits_idx] * filt_rechits_deposit * dep_scaling) / dep_energy_part
#             true_y = np.sum(rechit_y[filt_rechits_idx] * filt_rechits_deposit * dep_scaling) / dep_energy_part
#             true_z = np.sum(rechit_z[filt_rechits_idx] * filt_rechits_deposit * dep_scaling) / dep_energy_part
#
#
#             deposit_max = truth_rechit_deposit_max[filt_rechits_idx]
#             second_filter = np.greater(filt_rechits_deposit, deposit_max)
#
#             filt_rechits_idx = filt_rechits_idx[second_filter]
#             filt_rechits_deposit = filt_rechits_deposit[second_filter]
#             truth_rechit_deposit_max[filt_rechits_idx] = filt_rechits_deposit
#
#             truth_assignment[filt_rechits_idx] = overall_particle_id
#             overall_particle_id += 1
#
#             pdgid = simulation['particles_pdgid'][i]
#             truth_assignment_vertex_position_x[filt_rechits_idx] = simulation['particles_vertex_position_x'][i]
#             truth_assignment_vertex_position_y[filt_rechits_idx] = simulation['particles_vertex_position_y'][i]
#             truth_assignment_vertex_position_z[filt_rechits_idx] = simulation['particles_vertex_position_z'][i]
#             truth_assignment_x[filt_rechits_idx] = true_x
#             truth_assignment_y[filt_rechits_idx] = true_y
#             truth_assignment_z[filt_rechits_idx] = true_z
#             truth_assignment_momentum_direction_x[filt_rechits_idx] = simulation['particles_momentum_direction_x'][i]
#             truth_assignment_momentum_direction_y[filt_rechits_idx] = simulation['particles_momentum_direction_y'][i]
#             truth_assignment_momentum_direction_z[filt_rechits_idx] = simulation['particles_momentum_direction_z'][i]
#             truth_assignment_pdgid[filt_rechits_idx] = pdgid
#
#             # Set energy of muons to dep energy.
#             assigned_energy = simulation['particles_kinetic_energy'][i] if pdgid != 13 else dep_energy_part
#
#             truth_assignment_energy[filt_rechits_idx] = assigned_energy
#             truth_assignment_energy_dep[filt_rechits_idx] = dep_energy_part
#             truth_assignment_energy_dep_all[filt_rechits_idx] = simulation['particles_total_energy_deposited_all'][i]
#
#
#     if main_particle_simulation is not None:
#         add(main_particle_simulation)
#     for simulation in pu_simulations:
#         add(simulation)
#
#     # Convert to fraction
#     with np.errstate(divide='ignore', invalid='ignore'):
#         fraction_deposit_max = truth_rechit_deposit_max / rechit_energy
#     fraction_deposit_max[np.isnan(fraction_deposit_max)] = 0
#
#     if cut>0:
#         energy_by_area = rechit_energy / (rechit_area_norm if area_normed_cut else 1.0)
#         cut_off = energy_by_area < cut
#         rechit_energy[cut_off] = 0.0
#         truth_assignment[rechit_energy == 0.] = -1
#
#     rechit_energy = rechit_energy * rechit_scale
#
#     uniques, counts = np.unique(truth_assignment, return_counts=True)
#     counts = counts[uniques!=-1]
#     uniques = uniques[uniques!=-1]
#
#     if num_hits_cut >= 1:
#         for u, c in zip(uniques, counts):
#             if u == -1:
#                 continue
#             if c <= num_hits_cut:
#                 truth_assignment[truth_assignment==u] = -1
#
#     # truth_assignment_energy_dep = truth_assignment_energy_dep * 0.
#     truth_assignment_energy_dep_f = truth_assignment_energy_dep * 0.
#
#
#     # for u in uniques:
#     #     filt = truth_assignment==u
#     #     truth_assignment_energy_dep[filt] = np.sum(rechit_energy[filt] * fraction_deposit_max[filt])
#         # x_pos = np.sum(rechit_x[filt] * rechit_energy[filt] * fraction_deposit_max[filt]) / np.sum(rechit_energy[filt] * fraction_deposit_max[filt])
#         # y_pos = np.sum(rechit_y[filt] * rechit_energy[filt] * fraction_deposit_max[filt]) / np.sum(rechit_energy[filt] * fraction_deposit_max[filt])
#         # z_pos = np.sum(rechit_z[filt] * rechit_energy[filt] * fraction_deposit_max[filt]) / np.sum(rechit_energy[filt] * fraction_deposit_max[filt])
#         # truth_assignment_x[filt] = x_pos
#         # truth_assignment_y[filt] = y_pos
#         # truth_assignment_z[filt] = z_pos
#         # truth_assignment_energy_dep_f[filt] = np.sum(rechit_energy[filt])
#
#     truth_data = [truth_rechit_deposit_max,
#         truth_assignment_vertex_position_x,
#         truth_assignment_vertex_position_y,
#         truth_assignment_vertex_position_z,
#         truth_assignment_x,
#         truth_assignment_y,
#         truth_assignment_z,
#         truth_assignment_momentum_direction_x,
#         truth_assignment_momentum_direction_y,
#         truth_assignment_momentum_direction_z,
#         truth_assignment_pdgid,
#         truth_assignment_energy,
#         truth_assignment_energy_dep,
#         truth_assignment_energy_dep_f,
#         truth_assignment_energy_dep_all,]
#
#     for t in truth_data:
#         t[truth_assignment==-1] = 0
#
#     result = {
#         'rechit_x': rechit_x/10.,
#         'rechit_y': rechit_y/10.,
#         'rechit_z': rechit_z/10.,
#         'rechit_layer': rechit_layer,
#         'rechit_area': rechit_area/100.,
#         'rechit_energy': rechit_energy,
#         'truth_assignment': truth_assignment,
#         'truth_assignment_hit_dep': truth_rechit_deposit_max,
#         'truth_assignment_vertex_position_x': truth_assignment_vertex_position_x/10.,
#         'truth_assignment_vertex_position_y': truth_assignment_vertex_position_y/10.,
#         'truth_assignment_vertex_position_z': truth_assignment_vertex_position_z/10.,
#         'truth_assignment_x': truth_assignment_x/10.,
#         'truth_assignment_y': truth_assignment_y/10.,
#         'truth_assignment_z': truth_assignment_z/10.,
#         'truth_assignment_momentum_direction_x': truth_assignment_momentum_direction_x/10.,
#         'truth_assignment_momentum_direction_y': truth_assignment_momentum_direction_y/10.,
#         'truth_assignment_momentum_direction_z': truth_assignment_momentum_direction_z/10.,
#         'truth_assignment_pdgid': truth_assignment_pdgid,
#         'truth_assignment_energy': truth_assignment_energy,
#         'truth_assignment_energy_dep': truth_assignment_energy_dep,
#         'truth_assignment_energy_dep_f': truth_assignment_energy_dep_f,
#         'truth_assignment_energy_dep_all': truth_assignment_energy_dep_all,
#     }
#
#     if reduce:
#         filt = rechit_energy > 0
#         result = {k:v[filt] for k,v in result.items()}
#
#     return result

