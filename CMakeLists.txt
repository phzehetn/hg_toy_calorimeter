# $Id: CMakeLists.txt 104871 2017-06-23 14:17:13Z gcosmo $

#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
MESSAGE(${CMAKE_VERSION})
project(B4a)

find_package(Geant4 REQUIRED ui_all vis_all)

include(${Geant4_USE_FILE})
include_directories(jsoncpp/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/root/)
include_directories(/usr/local/root/)
include_directories(~/work/miniconda3x/include/c++/v1/)
include_directories(${Geant4_INCLUDE_DIRS})

include_directories(/usr/local/include/fastjet)
#include_directories(/afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/fastjet-install/include)

include_directories(../pythia8306/include)

add_subdirectory(pybind11)


set(CoreFiles src/Calorimeter.cc
        include/Calorimeter.hh
        src/PrimariesGenerator.cc
        include/PrimariesGenerator.hh
        src/B4DetectorConstruction.cc
        src/B4aActionInitialization.cc
        src/B4aSteppingAction.cc
        src/B4PartGeneratorBase.cc
        src/B4RunAction.cc
        src/B4aEventAction.cc
        src/SensorContainer.cc
        )

pybind11_add_module(minicalo src/minicalo_library.cc ${CoreFiles})
add_executable(test_cases_library test/test_cases_library.cc ${CoreFiles})


add_subdirectory(jsoncpp)


set(important_libraries ${Geant4_LIBRARIES} jsoncpp_lib /Users/shahrukhqasim/Workspace/NextCal/pythia8306/lib/libpythia8.dylib /usr/local/lib/libfastjet.dylib)
#set(important_libraries ${Geant4_LIBRARIES} jsoncpp_lib /afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/pythia8306/lib/libpythia8.so /afs/cern.ch/work/s/sqasim/workspace_phd_5/NextCal/fastjet-install/lib/libfastjet.so)


message( ${Geant4_LIBRARIES})
target_link_libraries(minicalo PRIVATE ${important_libraries})
target_link_libraries(test_cases_library PRIVATE pybind11::embed ${important_libraries})

#target_link_libraries(json_test ${JsonCppLib} ${Geant4_LIBRARIES})
#----------------------------------------------------------------------------
# Copy all scripts to the build directory, i.e. the directory in which we
# build B4a. This is so that we can run the executable directly because it
# relies on these scripts being in the current working directory.
#
set(EXAMPLEB4A_SCRIPTS
  exampleB4a.out
  exampleB4.in
  gui.mac
  init_vis.mac
  plotHisto.C
  run1.mac
  run2.mac
  vis.mac
  batchrun.mac
  condorSubmit
  batchscript.sh
  env.sh
  )

foreach(_script ${EXAMPLEB4A_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()
