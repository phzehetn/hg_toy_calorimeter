import numpy as np
import time
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import tensorflow as tf

@njit
def _merge_hits(particle_id, deposits, sensor_idx):
    # sensor_idx has to be sorted
    particle_tracking = particle_id[0]
    sensor_tracking = sensor_idx[0]
    total_sensor_dep = 0.0

    sensor_idx_result = np.zeros_like(sensor_idx) - 1
    deposit_result = np.zeros_like(deposits) - 1
    particle_id_result = np.zeros_like(particle_id) - 1
    last = 0

    for i in range(len(deposits)):
        if i!=0:
            if sensor_idx[i] < sensor_idx[i-1]:
                raise RuntimeError('Alignment issue')

        if not (sensor_idx[i] == sensor_tracking and particle_id[i] == particle_tracking):
            # Accumulate data
            particle_id_result[last] = particle_tracking
            sensor_idx_result[last] = sensor_tracking
            deposit_result[last] = total_sensor_dep
            last += 1

            # Reset state
            sensor_tracking = sensor_idx[i]
            particle_tracking = particle_id[i]
            total_sensor_dep = 0.0

        total_sensor_dep += deposits[i]

    sensor_idx_result[last] = sensor_tracking
    particle_id_result[last] = particle_tracking
    deposit_result[last] = total_sensor_dep

    return particle_id_result, sensor_idx_result, deposit_result

def merge_hits(particle_idx, sensor_idx, deposit):
    sort_idx = np.argsort(particle_idx)
    particle_idx_s = particle_idx[sort_idx]
    deposit_s = deposit[sort_idx]
    sensor_idx_s = sensor_idx[sort_idx]

    sort_idx = np.argsort(sensor_idx_s)
    particle_idx_s = particle_idx_s[sort_idx]
    deposit_s = deposit_s[sort_idx]
    sensor_idx_s = sensor_idx_s[sort_idx]


    particle_id_result, sensor_idx_result, deposit_result = _merge_hits(particle_idx_s, deposit_s, sensor_idx_s)
    filt = sensor_idx_result != -1
    particle_id_result = particle_id_result[filt]
    sensor_idx_result = sensor_idx_result[filt]
    deposit_result = deposit_result[filt]

    return particle_id_result, sensor_idx_result, deposit_result




# def merge_with_labels(particle_id, deposit, sensor_idx, label):
#     t1 = time.time()
#
#     sorting_idx = tf.argsort(sensor_idx)[..., tf.newaxis]
#     particle_id = tf.gather_nd(particle_id, sorting_idx)
#     deposit = tf.gather_nd(deposit, sorting_idx)
#     sensor_idx = tf.gather_nd(sensor_idx, sorting_idx)
#
#     sorting_idx = tf.argsort(particle_id)[..., tf.newaxis]
#     particle_id = tf.gather_nd(particle_id, sorting_idx)
#     deposit = tf.gather_nd(deposit, sorting_idx)
#     sensor_idx = tf.gather_nd(sensor_idx, sorting_idx)
#
#     row_splits = tf.ragged.segment_ids_to_row_splits(particle_id)
#     particle_id = tf.RaggedTensor.from_row_splits(particle_id, row_splits)
#     deposit = tf.RaggedTensor.from_row_splits(deposit, row_splits)
#     sensor_idx = tf.RaggedTensor.from_row_splits(sensor_idx, row_splits)
#
#     segment_ids = tf.ragged.row_splits_to_segment_ids(row_splits)
#     segment_sizes = row_splits[1:] - row_splits[0:-1]
#
#     merging_idx = []
#     splits = [0]
#     max_ = 0
#     for u in np.unique(label):
#         x = np.argwhere(u==label)[:, 0].tolist()
#         merging_idx += x
#         max_ += len(x)
#         splits.append(max_)
#     merging_idx = tf.convert_to_tensor(merging_idx)
#
#     reshuffled_sizes = tf.gather_nd(segment_sizes, merging_idx[..., tf.newaxis])
#     reshuffled_rs = tf.concat(([0], tf.cumsum(reshuffled_sizes)), axis=0)
#     reshuffled_si = tf.ragged.row_splits_to_segment_ids(reshuffled_rs)
#     x = tf.gather_nd(merging_idx, reshuffled_si[..., tf.newaxis])
#     x = tf.cast(x, tf.int64)
#
#     # x = tf.RaggedTensor.from_row_splits(x, reshuffled_rs)
#     y = tf.range(len(x), dtype=tf.int64) - tf.gather_nd(reshuffled_rs, reshuffled_si[..., tf.newaxis])
#     # print(x)
#     # print(tf.RaggedTensor.from_row_splits(y, reshuffled_rs))
#
#     tf.gather_nd(particle_id, tf.concat((x[..., tf.newaxis],y[..., tf.newaxis]), axis=-1))
#
#
#     # x = tf.range(len(x), dtype=tf.int64) - tf.RaggedTensor.from_row_splits(x, reshuffled_rs).row_splits
#
#     print("Took", time.time()-t1)
#     0/0

def intersection_deposits(particle_id, deposit, sensor_idx):
    """
    Computes a matrix where (i,j)th element of the matrix represents deposit of particle j on the sensors where particle
    i left some energy. The deposit is weighted according to sensor ith energy fraction on that sensor.

    :param particle_id:
    :param deposit:
    :param sensor_idx:
    :return:
    """
    @njit
    def operate(particle_id, deposit, sensor_idx, particle_total_deposit):
        # sensor_idx has to be sorted
        start = 0
        sensor_tracking = sensor_idx[0]
        total_sensor_dep = 0.0

        share = np.zeros_like(particle_total_deposit)
        share_x_particles = np.zeros((len(share), len(share)))

        # TODO: Check correct working at the end
        for i in range(len(deposit)+1):
            if sensor_idx[i] != sensor_tracking:
                # Accumulate data
                for j in range(start, i):
                    share[particle_id[j]] += deposit[j] * (deposit[j] / total_sensor_dep) / particle_total_deposit[particle_id[j]]

                    for k in range(start, i):
                        share_x_particles[particle_id[j],particle_id[k]] += deposit[j] * (deposit[k] / total_sensor_dep) / particle_total_deposit[particle_id[j]]

                # Reset state
                start = i
                sensor_tracking = sensor_idx[i]
                total_sensor_dep = 0.0

            total_sensor_dep += deposit[i]

        for i in range(len(particle_total_deposit)):
            share_x_particles[i,i] = 0.0

        return share, share_x_particles

    @njit
    def compute_total_dep(particle_id, deposit):
        P = np.unique(particle_id)
        sensor_weights = np.zeros(len(P))
        for p in P:
            filt = particle_id==p
            dep = deposit[filt]
            total_energy = np.sum(dep)
            sensor_weights[p] = total_energy
        return sensor_weights

    sort_idx = np.argsort(sensor_idx)

    particle_id_s = particle_id[sort_idx]
    deposit_s = deposit[sort_idx]
    sensor_idx_s = sensor_idx[sort_idx]

    pid_max = np.max(particle_id)

    total_deposits = compute_total_dep(particle_id_s, deposit_s)
    share, share_x_particles = operate(particle_id_s, deposit_s, sensor_idx_s, total_deposits)

    # print(share - np.sum(share_x_particles, axis=1))

    #
    # print("MINMAX", np.min(share), np.max(share))
    # plt.hist(share.flatten(), histtype='step')
    # plt.hist(1 - np.sum(share_x_particles, axis=1), histtype='step')
    # plt.ylabel('Num particles')
    # plt.xlabel('sum_i (sum_j(deposit of particle i over sensor j * share of energy over sensor j))  / total deposit')
    # plt.show()

    # f = share<0.94
    # x = np.max(share_x_particles, axis=1)
    # print("MINMAX", np.min(x[f]), np.max(x[f]))
    # plt.hist(x[f].flatten())
    # # plt.yscale('log')
    # plt.show()

    return share, share_x_particles
        # print("Took", time.time()-t1,"seconds")

    # 0/0

    # df = pd.DataFrame({
    #     'particle_id' : particle_id_s,
    #     'deposit' : deposit_s,
    #     'sensor_idx' : sensor_idx_s,
    # })
    #
    # df.to_csv('../random_data/share_computation_raw_data.csv')
    # 0/0



if __name__ == '__main__':
    particle_idx = [5,5,5,3,3,3,2,2,2,1,1,1]
    sensor_idx =   [0,0,1,0,0,1,2,2,1,0,0,0]
    deposit =      [1,1,1,1,1,1,1,1,1,1,1,1]

    particle_idx = np.array(particle_idx)
    sensor_idx = np.array(sensor_idx)
    deposit = np.array(deposit)

    p2, s2, d2 = merge_hits(particle_idx, sensor_idx, deposit)

    print(p2, s2, d2)


