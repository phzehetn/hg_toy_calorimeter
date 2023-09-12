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

E_dep, E_true = [], []
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
		masknoise = truth['truthHitAssignementIdx'] != -1
		masktrack = feature['recHitZ'] != 315
		mask = np.logical_and(masknoise, masktrack)
		e_dep = np.sum(feature['recHitEnergy'][mask])
		e_true = truth['truthHitAssignedEnergies'][mask][0]
		E_dep.append(e_dep)
		E_true.append(e_true)

E_dep = np.array(E_dep)
E_true = np.array(E_true)
ratio = E_dep / E_true
print(ratio)
data = (E_true, E_dep)
outfile = os.path.join(directory, 'resolution.pkl')
with open(outfile, 'wb') as f:
	pickle.dump(data, f)
		
