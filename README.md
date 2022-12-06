## Warning
Not ready yet. Use with caution.

# A High Granularity Toy Calorimeter
The source code for a toy calorimeter to study particle reconstruction performance.

This calorimeter was also employed in the following paper:
https://link.springer.com/article/10.1140/epjc/s10052-022-10665-7


Cite as:
```
@article{Qasim_2022,
	doi = {10.1140/epjc/s10052-022-10665-7},
	url = {https://doi.org/10.1140%2Fepjc%2Fs10052-022-10665-7},
	year = 2022,
	month = {aug},
	publisher = {Springer Science and Business Media {LLC}},
	volume = {82},
	number = {8},
	author = {Shah Rukh Qasim and Nadezda Chernyavskaya and Jan Kieseler and Kenneth Long and Oleksandr Viazlo and Maurizio Pierini and Raheel Nawaz},
	title = {End-to-end multi-particle reconstruction in high occupancy imaging calorimeters with graph neural networks},
	journal = {The European Physical Journal C}
}
```

## Install / Compile
The following docker container should be used to run the code:
https://hub.docker.com/r/shahrukhqasim2/g4calo

The singularity .def and Dockerfile are available in `containers/` directory if the container needs
to be built.

```
singularity pull docker://shahrukhqasim2/g4calo:latest
```
A `.sif` file must have been created in the current directory. Login to the `.sif` container:
```
singularity shell g4calo.sif
```
If at CERN, `EOS` and `AFS` can also be mounted as follows:
```
singularity shell -B /afs -B /eos g4calo.sif
```
Check out the repository:
```
git clone https://github.com/shahrukhqasim/hg_toy_calorimeter.git
cd hg_toy_calorimeter
git submodule update --init --recursive
```

And build:
```
mkdir build
cd build
cmake ..
make -j
cd ..
```


## Generate simulations
First the build directory should be added `PYTHONPATH` and then
`run_simulation.py` script can be used to generate a set of events:
```
cd python/bin
export PYTHONPATH=$PYTHONPATH:`readlink -f ../../build/`:`readlink -f ../../build/lib/`:`readlink -f ../libs`
python3 run_simulations.py minbias --cores=1
```
The simulations will be generated in `ra_pickles` format, as described here:
https://github.com/shahrukhqasim/ra_pickles

They are inherently stored as pickled numpy arrays and therefore, can be accessed
outside of the container, by installing the `ra_pickles` package via
`pip3 install ra-pickles`. The container should only be used for simulations and
event generations.

## Generate events
Hello world