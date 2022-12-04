//
// Created by Shah Rukh Qasim on 07.01.22.
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>


#ifndef B4A_CALORIMETER_HH
#define B4A_CALORIMETER_HH

void initialize(std::string detector_specs, std::string pythiadata, bool collect_full_data, int rseed_0, int rseed_1, int rseed_2, int rseed_3);
void initialize_test(long rseed, std::string detector_specs);
void wrap_up();
pybind11::dict simulate_pu();
pybind11::dict generate_pu_without_sim();
pybind11::dict simulate_qqbar2ttbar();

pybind11::dict simulate_particle(double position_x, double position_y, double position_z, double direction_x,
                           double direction_y,
                           double direction_z, int pdgid, double energy);

pybind11::dict dict_check(int number_of_events, long rseed, std::string detector_specs);
pybind11::dict get_sensor_data();
#endif //B4A_CALORIMETER_HH
