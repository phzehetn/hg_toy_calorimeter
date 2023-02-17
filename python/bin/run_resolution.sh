#!/bin/bash

echo "Starting with Photons"
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_100GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_150GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_150GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_200GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_500GeV_eta20_sp/dataCollection.djcdc

echo "Starting with Pions"
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_20GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_20GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_50GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_50GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_100GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_100GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_150GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_150GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_200GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_200GeV_eta20_sp/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_500GeV_eta20_spf/dataCollection.djcdc
python3 resolution.py /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/pions_500GeV_eta20_sp/dataCollection.djcdc

echo "Done"
