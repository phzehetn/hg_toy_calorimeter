import os
import sys
import pickle
import numpy as np

if len(sys.argv) < 2:
	print("Provide output path")
	exit(1)

file_list = [
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_150GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_150GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_20GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_20GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_50GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_50GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_100GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_100GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_150GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_150GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_200GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_200GeV_eta20_sp/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_500GeV_eta20_spf/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_500GeV_eta20_sp/calibration.pkl",
]

summary = {}

for fname in file_list:
	if not os.path.exists(fname):
		print(fname, " not found")
	else:
		path, _ = os.path.split(fname)
		_, id = os.path.split(path)
		with open(fname, 'rb') as f:
			data = pickle.load(f)
		summary[id] = data

with open(sys.argv[1], 'wb') as f:
	pickle.dump(summary, f)
