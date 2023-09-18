import os
import sys
import gzip
import pickle
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator


if len(sys.argv) < 3:
    print('Please provide path to dataCollection as first argument\
            and output path as second argument')
    exit(1)

if len(sys.argv) > 3:
    print('Found more than two command line arguments, \
            only the first two will be used')
path = sys.argv[1]
outpath = sys.argv[2]
assert os.path.exists(path)
if os.path.exists(outpath):
    print("Ouput file already exists, will be overwritten")

dc = DataCollection(path)
files = [dc.dataDir + sample for sample in dc.samples]
N = len(files)
print("Using dataCollection: \n", path)
print("Found ", N, " files")
print("Only using first file")

td = dc.dataclass()
td.readFromFileBuffered(files[0])

gen = TrainDataGenerator()
gen.setBatchSize(1)
gen.setSquaredElementsLimit(False)
gen.setSkipTooLargeBatches(False)
gen.setBuffer(td)

num_steps = gen.getNBatches()
print(num_steps, " events found in first file")
generator = gen.feedNumpyData()

features = []
truths = []
for i in range(num_steps):
    data = next(generator)[0]
    features.append(td.createFeatureDict(data))
    truths.append(td.createTruthDict(data))

outdict = {
        'features': features,
        'truths': truths,
        }

with gzip.open(outpath, 'wb') as f:
    pickle.dump(outdict, f)
print("DONE")
