import gzip
import pickle
import ra_pickles
from calo.event_generator import EventGenerator
import numpy as np
from plotting.plotting_3d import to_plotly
import os

# TODO: Change file/folder names to be more generic

calib_path = os.getenv('HG_TOY_CALORIMETER_CALIBRATION_DATA')

with gzip.open(calib_path, 'rb') as f:
    sensor_data = pickle.load(f)

# 5.2e-5, 0.006

noise_fluctuations = ('type_a', 0,1.2e-5)
gen = EventGenerator(sensor_data, noise_fluctuations, cut=0.0009, num_hits_cut=3, reduce=True, area_normed_cut=True, merge_closeby_particles=False)

em_path = '/Users/shahrukhqasim/Workspace/NextCal/hg_toy_calorimeter/data/calibration/em'
had_path = '/Users/shahrukhqasim/Workspace/NextCal/hg_toy_calorimeter/data/calibration/had'

em_reader = ra_pickles.RandomAccessPicklesReader(em_path)
had_reader = ra_pickles.RandomAccessPicklesReader(had_path)

print(em_reader.get_element(0)[0].keys())

the_sim = em_reader.get_element(0)

for i in range(10):
    gen.add([em_reader.get_element(i)[0]], minbias=False)
for i in range(10):
    gen.add([had_reader.get_element(i)[0]], minbias=False)

# gen.add()
d = gen.process()

rechit_energy = d['rechit_energy']

print(np.sum(rechit_energy[d['truth_assignment']==-1]))
print(np.sum(d['truth_assignment'][rechit_energy>0]==-1))

to_plotly(d, 'x.html', stringy=True)
