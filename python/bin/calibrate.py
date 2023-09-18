import pdb
import json
import os
import sys
import pickle
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ra_pickles
from copy import deepcopy
from scipy.optimize import curve_fit

BINS = 50
N_SIGMA = 3

def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


##############################################################################################
### Load Data ################################################################################
##############################################################################################

assert len(sys.argv) == 2, "Please provide a cut value as argument"
noise_cut = np.float64(sys.argv[1])

if not os.path.exists('summary.json'):
    summary = {}
else:
    with open('summary.json', 'r') as f:
        summary = json.load(f)

em_path = '/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/em'
had_path = '/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/had'
sensor_path = '/afs/cern.ch/user/p/phzehetn/Code/HGTOY/sensor_data_calibrated.bin.gz'

print("Start Loading Data: ")
print("EM: ", em_path)
print("HAD: ", had_path)
print("Sensor: ", sensor_path)
print("Cut value: ", noise_cut)

em_reader = ra_pickles.RandomAccessPicklesReader(em_path)
had_reader = ra_pickles.RandomAccessPicklesReader(had_path)
em_sims = [em_reader.get_element(i) for i in range(em_reader.get_total())]
had_sims = [had_reader.get_element(i) for i in range(had_reader.get_total())]

with gzip.open(sensor_path, 'rb') as f:
    sensor_data = pickle.load(f)
sensor_scaling = 1 + sensor_data['sensors_pre_absorber_thickness'] / sensor_data['sensors_thickness']
sensor_data['sensors_scaling'] = sensor_scaling
normed_area = sensor_data['sensors_area'] / np.max(sensor_data['sensors_area'])

em_rechit_idx = np.concatenate([x[0]['rechit_idx'] for x in em_sims])
em_rechit_energy = np.concatenate([x[0]['rechit_energy'] for x in em_sims])
em_rechit_ev = np.concatenate([(x[0]['rechit_energy'] * 0 + i).astype(np.int32) for i, x in enumerate(em_sims)])
em_rechit_weights = tf.gather_nd(sensor_data['sensors_scaling'], em_rechit_idx[..., tf.newaxis])
em_rechit_normed_area = tf.gather_nd(normed_area, em_rechit_idx[..., tf.newaxis])
em_rechit_is_zero = em_rechit_energy / em_rechit_normed_area < noise_cut
em_rechit_energy[em_rechit_is_zero] = 0.0
em_dep_energy = tf.math.segment_sum(em_rechit_energy * em_rechit_weights, em_rechit_ev)
em_true_energy = np.array([np.sum(x[1]['energy']) for x in em_sims])

had_rechit_idx = np.concatenate([x[0]['rechit_idx'] for x in had_sims])
had_rechit_energy = np.concatenate([x[0]['rechit_energy'] for x in had_sims])
had_rechit_ev = np.concatenate([(x[0]['rechit_energy'] * 0 + i).astype(np.int32) for i, x in enumerate(had_sims)])
had_rechit_weights = tf.gather_nd(sensor_data['sensors_scaling'], had_rechit_idx[..., tf.newaxis])
had_rechit_normed_area = tf.gather_nd(normed_area, had_rechit_idx[..., tf.newaxis])
had_rechit_is_zero = had_rechit_energy / had_rechit_normed_area < noise_cut
had_rechit_energy[had_rechit_is_zero] = 0.0
had_dep_energy = tf.math.segment_sum(had_rechit_energy * had_rechit_weights, had_rechit_ev)
had_true_energy = np.array([np.sum(x[1]['energy']) for x in had_sims])

##############################################################################################
### EM Calibration ###########################################################################
##############################################################################################

print("Starting EM Calibration")


fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].set_title("EM events, before calibration", fontsize=20)
ratio = np.array(em_dep_energy / em_true_energy)
n_em, bins_em, _ = ax[0].hist(ratio, bins=BINS, label='EM events')
ax[0].set_xlabel('Deposited Energy / True Energy', fontsize=20)
ax[0].set_ylabel('Events', fontsize=20)
ax[0].tick_params(axis='both', labelsize=20)

bin_centers = 0.5*(bins_em[1:] + bins_em[:-1])
p0 = np.array([1., 0.1, 1.])
coeff, var_matrix = curve_fit(gauss, bin_centers, n_em, p0=p0)
hist_fit = gauss(bin_centers, *coeff)
label = "Gaussian fit: \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
ax[0].plot(bin_centers, hist_fit, 'r-', linewidth=2, label=label)
# mask events outside of N_SIGMA sigma
mask = np.abs(bin_centers - coeff[1]) < N_SIGMA * np.abs(coeff[2])
# # fit again a gaussian to masked events
coeff_masked, var_matrix_masked = curve_fit(gauss, bin_centers[mask], n_em[mask], p0=coeff)
g_em = 1 / coeff_masked[1]
hist_fit_masked = gauss(bin_centers, *coeff)
label = "Gaussian fit with " + r"$3\, \sigma$" + " mask \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
ax[0].plot(bin_centers, hist_fit, 'g-', linewidth=2, label=label)
ax[0].legend(fontsize=20)
# add textbox with masked info
textstr = f"{np.sum(mask)} outliers out of {len(mask)} events masked\n"
textstr += r"$g_{EM}$ = %.3f" % g_em
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax[0].text(0.5, 0.5, textstr, transform=ax[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


### Perform EM calibration

sensor_data_em = deepcopy(sensor_data)
sensor_data_em['sensors_scaling'][sensor_data_em['sensors_active_layer_num'] < 28] *= g_em
em_rechit_weights_em = tf.gather_nd(sensor_data_em['sensors_scaling'], em_rechit_idx[..., tf.newaxis])
em_dep_energy_em = tf.math.segment_sum(em_rechit_energy * em_rechit_weights_em, em_rechit_ev)


ax[1].set_title("EM events, after correcting for mean", fontsize=20)
ratio = np.array(em_dep_energy_em / em_true_energy)
n_em_calib, bins_em_calib, _ = ax[1].hist(ratio, bins=BINS, label='EM events', range=(0.7, 1.3))
# fit histogram
bin_centers = 0.5*(bins_em_calib[1:] + bins_em_calib[:-1])
p0 = [1., 0., 1.]
coeff, var_matrix = curve_fit(gauss, bin_centers, n_em_calib, p0=p0)
hist_fit = gauss(bin_centers, *coeff)
label = "Gaussian fit: \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
label = "Gaussian fit: \n" + r"$\mu$ = %.3f" % (coeff[1]) + "\n" + r"$\sigma$ = %.3f" % (coeff[2])
ax[1].plot(bin_centers, hist_fit, 'r-', linewidth=2, label=label)

ax[1].set_xlabel('Calibrated EM Energy / True Energy', fontsize=20)
ax[1].set_ylabel('Events', fontsize=20)
ax[1].tick_params(axis='both', labelsize=20)
ax[1].legend(fontsize=20)
fig.savefig(f"cut_{noise_cut}_em_calibration.png")
print(f"EM Calibration done, g_EM = {g_em}")


##############################################################################################
### Hadronic Calibration #####################################################################
##############################################################################################

print("Starting Hadronic Calibration")
# Sensor with EM part calibrated and hadronic part shut off
sensor_data_no_had = deepcopy(sensor_data_em)
sensor_data_no_had['sensors_scaling'][sensor_data_no_had['sensors_active_layer_num'] >= 28] = 0.0

# Sensor with only hadronic part and EM part shut off
sensor_data_no_em = deepcopy(sensor_data)
sensor_data_no_em['sensors_scaling'][sensor_data_no_em['sensors_active_layer_num'] < 28] = 0.0
had_rechit_weights_no_had = tf.gather_nd(sensor_data_no_had['sensors_scaling'], had_rechit_idx[..., tf.newaxis])
had_dep_energy_no_had = tf.math.segment_sum(had_rechit_energy * had_rechit_weights_no_had, had_rechit_ev)
had_rechit_weights_no_em = tf.gather_nd(sensor_data_no_em['sensors_scaling'], had_rechit_idx[..., tf.newaxis])
had_dep_energy_no_em = tf.math.segment_sum(had_rechit_energy * had_rechit_weights_no_em, had_rechit_ev)
# True energy minus the part deposited in the EM part. I.e. true energy of the hadronic part
had_true_calib = had_true_energy - had_dep_energy_no_had

fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].set_title("Hadronic events, before calibration", fontsize=20)
ratio = np.array(had_dep_energy_no_em / had_true_calib)
n_had, bins_had, _ = ax[0].hist(ratio, bins=BINS, label='Hadronic events', range=(-0.25, 1))
ax[0].set_xlabel('Deposited Energy / (True Energy - Energy lost in EM)', fontsize=20)
ax[0].set_ylabel('Events', fontsize=20)

# fit histogram with gaussian
bin_centers = 0.5*(bins_had[1:] + bins_had[:-1])
p0 = [1., 0., 1.]
coeff, var_matrix = curve_fit(gauss, bin_centers, n_had, p0=p0)
hist_fit = gauss(bin_centers, *coeff)
label = "Gaussian fit: \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
ax[0].plot(bin_centers, hist_fit, 'r-', linewidth=2, label=label)

# mask events outside of N_SIGMA sigma
mask = np.abs(bin_centers - coeff[1]) < N_SIGMA * np.abs(coeff[2])
# # fit again a gaussian to masked events
coeff_masked, var_matrix_masked = curve_fit(gauss, bin_centers[mask], n_had[mask], p0=coeff)
g_had = 1 / coeff_masked[1]
hist_fit_masked = gauss(bin_centers, *coeff)
label = "Gaussian fit with " + r"$3\, \sigma$" + " mask \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
ax[0].plot(bin_centers, hist_fit_masked, 'g-', linewidth=2, label=label)


sensor_data_calibrated = deepcopy(sensor_data_em)
sensor_data_calibrated['sensors_scaling'][sensor_data_calibrated['sensors_active_layer_num'] >= 28] *= g_had
had_rechit_weights_calibrated = tf.gather_nd(sensor_data_calibrated['sensors_scaling'], had_rechit_idx[..., tf.newaxis])
had_dep_energy_calibrated = tf.math.segment_sum(had_rechit_energy * had_rechit_weights_calibrated, had_rechit_ev)

ax[1].set_title("Hadronic events, after calibration", fontsize=20)
ratio = np.array(had_dep_energy_calibrated / had_true_energy)
n_had_calib, bins_had_calib, _ = ax[1].hist(
    ratio, 
    bins=BINS, label='Hadronic events', range=(0.0, 2.0))


# fit histogram with gaussian
bin_centers = 0.5*(bins_had_calib[1:] + bins_had_calib[:-1])
p0 = [1., 0., 1.]
coeff, var_matrix = curve_fit(gauss, bin_centers, n_had_calib, p0=p0)
hist_fit = gauss(bin_centers, *coeff)
label = "Gaussian fit: \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
ax[1].plot(bin_centers, hist_fit, 'r-', linewidth=2, label=label)

ax[1].set_xlabel('Calibrated Energy / True Energy', fontsize=20)


ax[0].legend(fontsize=15, loc='upper right')
ax[1].legend(fontsize=15)
fig.savefig(f"cut_{noise_cut}_had_calibration.png")

print(f"Hadronic Calibration done, g_had = {g_had}")

##############################################################################################
### Summary ###################################################################################
##############################################################################################


# sensor data calibrated
with gzip.open(f'noise_cut_{noise_cut}_sensor_data_calibrated.bin.gz', 'wb') as f:
    pickle.dump(sensor_data_calibrated, f)

weights_em = tf.gather_nd(sensor_data_calibrated['sensors_scaling'], em_rechit_idx[..., tf.newaxis])
weights_had = tf.gather_nd(sensor_data_calibrated['sensors_scaling'], had_rechit_idx[..., tf.newaxis])

em_hits = tf.math.segment_sum(em_rechit_energy * weights_em, em_rechit_ev)
had_hits = tf.math.segment_sum(had_rechit_energy * weights_had, had_rechit_ev)

fig, ax = plt.subplots(1, 2, figsize=(20,10))
fig.suptitle("After Calibration", fontsize=20)
ratio = np.array(em_hits / em_true_energy)
n_em, bins_em, _ = ax[0].hist(ratio, bins=BINS, label='EM events', range=(0.0, 2.0))
ax[0].set_title("EM", fontsize=20)
# fit histogram with gauss
bin_centers = 0.5*(bins_em[1:] + bins_em[:-1])
p0 = [1., 0., 1.]
coeff, var_matrix = curve_fit(gauss, bin_centers, n_em, p0=p0)
sigma_em = coeff[2]
hist_fit = gauss(bin_centers, *coeff)
label = "Gaussian fit: \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
ax[0].plot(bin_centers, hist_fit, 'r-', linewidth=2, label=label)
ax[0].legend(fontsize=15, loc='upper right')

ratio = np.array(had_hits / had_true_energy)
n_had, bins_had, _ = ax[1].hist(ratio, bins=BINS, label='Hadronic events', range=(0.0, 2.0))
ax[1].set_title("Hadronic", fontsize=20)
# fit histogram with gauss
bin_centers = 0.5*(bins_had[1:] + bins_had[:-1])
p0 = [1., 0., 1.]
coeff, var_matrix = curve_fit(gauss, bin_centers, n_had, p0=p0)
sigma_had = coeff[2]
hist_fit = gauss(bin_centers, *coeff)
label = "Gaussian fit: \n" + r"$\mu$ = %.3f, $\sigma$ = %.3f" % (coeff[1], coeff[2])
ax[1].plot(bin_centers, hist_fit, 'r-', linewidth=2, label=label)
ax[1].legend(fontsize=15, loc='upper right')

# set ticklabel size
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)

ax[0].set_xlabel('Calibrated Energy / True Energy', fontsize=20)
ax[1].set_xlabel('Calibrated Energy / True Energy', fontsize=20)
ax[0].set_ylabel('Number of events', fontsize=20)
ax[1].set_ylabel('Number of events', fontsize=20)

fig.savefig(f"cut_{noise_cut}_calibrated.png")
print(f"Calibration done, g_em = {g_em}, g_had = {g_had}")

summary[str(noise_cut)] = {
    'g_em': g_em,
    'g_had': g_had,
    'sigma_em': sigma_em,
    'sigma_had': sigma_had,
}
with open('summary.json', 'w') as f:
    json.dump(summary, f)
