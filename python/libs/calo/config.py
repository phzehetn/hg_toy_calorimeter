import os
import json

is_laptop = False
if os.path.exists('/Users/shahrukhqasim/Workspace'):
    is_laptop=True


is_fi = False
if os.path.exists('/mnt/home/sqasim/ceph/Workspace'):
    is_fi = True


if is_laptop:
    binary_path = '/bin/exampleB4a'
else:
    binary_path = '/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/miniCalo/build3/exampleB4a'



# if is_laptop:
#     detector_specs_file = '/scripts/toydetector/detector_specs.json'
# elif is_fi:
#     detector_specs_file = '/mnt/ceph/users/sqasim/Workspace/NextCal/ShahRukhStudies/scripts/toydetector/detector_specs.json'
# else:
#     detector_specs_file = '/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/ShahRukhStudies/scripts/toydetector/detector_specs.json'
#
# with open(detector_specs_file, 'r') as f:
#     detector_specs = json.load(f)


def set_env():
    if is_laptop:
        os.environ["G4PARTICLEXSDATA"]="/usr/local/share/Geant4-10.7.2/data/G4PARTICLEXS3.1.1"
        os.environ["G4LEDATA"]="/usr/local/share/Geant4-10.7.2/data/G4EMLOW7.13"
        os.environ["G4LEVELGAMMADATA"]="/usr/local/share/Geant4-10.7.2/data/PhotonEvaporation5.7"
        os.environ["G4ENSDFSTATEDATA"]="/usr/local/share/Geant4-10.7.2/data"
        os.environ["G4VIS_BUILD_OPENGLX_DRIVER"]="1"
    else:
        os.environ["G4PARTICLEXSDATA"]="/usr/local/geant4/10.7.1/share/Geant4-10.7.2/data/G4PARTICLEXS3.1.1"
        os.environ["G4LEDATA"]="/usr/local/geant4/10.7.1/share/Geant4-10.7.2/data/G4EMLOW7.13"
        os.environ["G4LEVELGAMMADATA"]="/usr/local/geant4/10.7.1/share/Geant4-10.7.2/data/PhotonEvaporation5.7"
        os.environ["G4ENSDFSTATEDATA"]="/usr/local/geant4/10.7.1/share/Geant4-10.7.2/data/G4ENSDFSTATE2.3"
        os.environ["G4VIS_BUILD_OPENGLX_DRIVER"]="1"

