# Script to calculate resolution of toy detector
import os
import sys
import pickle
import gc
from tqdm import tqdm
import numpy as np
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator

if len(sys.argv) == 0:
        print("Please provide path to DataCollection")
        exit(1)
else:
        dcpath = sys.argv[1]
assert os.path.exists(dcpath)

directory, dcname = os.path.split(dcpath)

print("Analysing: ", dcpath)
dc = DataCollection(dcpath)

N = []
N_noise = []
E_noise = []
for i in range(len(dc.samples)):
        td = dc.dataclass()
        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setSkipTooLargeBatches(False)
        gen.setSquaredElementsLimit(False)
        td.readFromFileBuffered(dc.dataDir + dc.samples[i])
        gen.setBuffer(td)

        n_samples = gen.getNBatches()
        generator = gen.feedNumpyData()

        print("Found ", str(n_samples), "events")
        for j in tqdm(range(n_samples)):
                data = next(generator)[0]
                truth = td.createTruthDict(data)
                feature = td.createFeatureDict(data)
                N.append(truth['truthHitAssignementIdx'].shape[0])
                noise = np.array(truth['truthHitAssignementIdx']).reshape(-1) 
                N_noise.append(np.sum(noise == -1))
                e = np.sum(feature['recHitEnergy'][noise == -1])
                E_noise.append(e)
print(N)
print("mean: ", np.mean(N))
print("NOISE:")
print(N_noise)
print("mean: ", np.mean(N_noise))
print(E_noise)
print("mean: ", np.mean(E_noise))
                
