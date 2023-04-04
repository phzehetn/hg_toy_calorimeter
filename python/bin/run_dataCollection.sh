#!/bin/bash

cd /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0009_PU2/
createDataCollectionFromTD.py -c TrainData_NanoML -o dataCollection.djcdc *.djctd

cd /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0010_PU2/
createDataCollectionFromTD.py -c TrainData_NanoML -o dataCollection.djcdc *.djctd

cd /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0011_PU2/
createDataCollectionFromTD.py -c TrainData_NanoML -o dataCollection.djcdc *.djctd

cd /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0012_PU2/
createDataCollectionFromTD.py -c TrainData_NanoML -o dataCollection.djcdc *.djctd

cd /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0013_PU2/
createDataCollectionFromTD.py -c TrainData_NanoML -o dataCollection.djcdc *.djctd

cd /eos/home-p/phzehetn/ML4Reco/Data/V4/calibration/Events/photons_50GeV_eta20_0014_PU2/
createDataCollectionFromTD.py -c TrainData_NanoML -o dataCollection.djcdc *.djctd
