import os
import sys
import pickle
import numpy as np

if len(sys.argv) < 2:
	print("Provide output path")
	exit(1)

file_list = [
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0009/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0009/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0009/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0009/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0010/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0010/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0010/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0010/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0011/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0011/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0011/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0011/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0012/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0012/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0012/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0012/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0013/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0013/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0013/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0013/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0014/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0014/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0014/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0014/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0015/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0015/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0015/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0015/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0016/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0016/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0016/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0016/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0017/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0017/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0017/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0017/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0018/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0018/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0018/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0018/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0019/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0019/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0019/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0019/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0020/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_0020/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_0020/calibration.pkl",
	"/eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_0020/calibration.pkl",
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
