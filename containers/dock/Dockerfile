# syntax=docker/dockerfile:1
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt update
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y dialog
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt update
RUN apt-get install -y qt5-default
RUN apt-get install -y libxerces-c-dev

RUN apt-get install -y gfortran
RUN apt-get install -y python3-dev python3-numpy-dev

RUN apt-get install -y libxpm-dev libxft-dev libxmu-dev
RUN apt-get install -y libssl-dev

RUN apt-get install -y libpcre3-dev xlibmesa-glu-dev libglew1.5-dev libftgl-dev libfftw3-dev libcfitsio-dev graphviz-dev libxml2-dev libgsl0-dev


RUN apt-get install  -y curl
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y htop


WORKDIR /
RUN curl -SLO https://fastjet.fr/repo/fastjet-3.4.0.tar.gz
RUN tar -xzvf fastjet-3.4.0.tar.gz
WORKDIR /fastjet-3.4.0
RUN ./configure --prefix=$PWD/../fastjet-install
RUN make -j
RUN make check
RUN make install

WORKDIR /
RUN curl -SLO https://pythia.org/download/pythia83/pythia8308.tgz
RUN tar -zxvf pythia8308.tgz
WORKDIR /pythia8308
RUN ./configure
RUN make -j

WORKDIR /
RUN curl -SLO https://root.cern/download/root_v6.22.08.Linux-ubuntu20-x86_64-gcc9.3.tar.gz
RUN tar zxf root_v6.22.08.Linux-ubuntu20-x86_64-gcc9.3.tar.gz -C /usr/local/
RUN chown -R `id -un`:`id -gn` /usr/local/root
RUN chmod +x /usr/local/root/bin/thisroot.*
RUN wget https://gitlab.cern.ch/geant4/geant4/-/archive/v10.7.2/geant4-v10.7.2.tar.gz
RUN tar -zxvf geant4-v10.7.2.tar.gz
RUN mkdir geant4-build
WORKDIR /geant4-build
RUN cmake  -DCMAKE_INSTALL_PREFIX=/usr/local/geant4/10.7.1 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGEANT4_BUILD_MULTITHREADED=ON -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_GDML=ON -DGEANT4_USE_QT=ON -DGEANT4_USE_OPENGL_X11=ON -DGEANT4_ENABLE_TESTING=OFF ../geant4-v10.7.2
RUN make -j4
RUN make install
RUN apt-get -y install python3-pip
RUN pip3 install networkx
RUN pip3 install torch torchvision torchaudio
RUN pip3 install awkward0
RUN pip3 install matplotlib
RUN pip3 install uproot3
RUN strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5
RUN pip3 install pandas scipy plotly tensorflow
RUN pip3 install ra-pickles
RUN pip3 install argh


ENV LISTEN_PORT=12345
ENV LC_ALL=C
ENV LD_PRELOAD=/geant4-build/BuildProducts/lib/libG4particles.so:/geant4-build/BuildProducts/lib/libG4processes.so