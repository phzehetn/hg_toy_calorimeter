import numpy as np
import tensorflow as tf


def x_y_z_to_eta_phi_theta(x, y, z):
    # phi = np.arctan(y / x)
    phi = np.arctan2(y, x)
    s = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(s, z)
    eta = - np.log(np.tan(theta / 2))

    return eta, phi, theta

def to_2pi_m2pi(x):
    return np.fmod(x, np.pi*2)


def angle_diff(x, y):
    return np.arctan2(np.sin(x - y), np.cos(x - y))


def angle_diff(x, y):
    """
    Returns symmetric difference in the range of [0,pi]
    """
    z = np.arctan2(np.sin(x - y), np.cos(x - y))
    z = np.abs(z)
    return z



def angle_diff_tf(x, y):
    """
    Returns symmetric difference in the range of [0,pi]
    """
    z = tf.math.atan2(tf.math.sin(x - y), tf.math.cos(x - y))
    z = tf.abs(z)
    return z


def eta_phi_z_to_x_y_z(eta, phi, z):
    theta = 2 * np.arctan(np.exp(-eta))
    s = z * np.tan(theta)

    x = np.sqrt(s ** 2 / (np.tan(phi) ** 2 + 1))
    x = -x if phi > np.pi else x
    y = x * np.tan(phi)

    return x, y, z


def eta_to_theta(eta):
    return  2 * np.arctan(np.exp(-eta))

def theta_to_eta(theta):
    return - np.log(np.tan(theta / 2))



def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between_vectors(v1, v2):
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def nxnAngles_tf(D1, D2):
    """
    Computes angles between every vector of D1 with every vector D2. Data axis is axis 1 (not axis 0).
    """
    D1 = D1 / tf.linalg.norm(D1, axis=1)[..., tf.newaxis]
    D2 = D2 / tf.linalg.norm(D2, axis=1)[..., tf.newaxis]

    x = tf.reduce_sum(D1[tf.newaxis, :, :] * D2[:, tf.newaxis, :], axis=-1)
    x = tf.clip_by_value(x, -1.0, +1.0)
    x = tf.acos(x)

    return x
