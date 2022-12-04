//
// Created by Shah Rukh Qasim on 07.01.22.
//
#include <pybind11/pybind11.h>
#include "Calorimeter.hh"

namespace py = pybind11;

float cpp_test_function(float a, float b) {
    return a*2 + b;
}


py::tuple cpp_test_function_2(float a, float b) {
    return py::make_tuple(a*2+1, a*3+b);
}


PYBIND11_MODULE(minicalo, handle){
    handle.doc() = "Mini calo library";
    handle.def("cpp_test_function", &cpp_test_function);
    handle.def("cpp_test_function_2", &cpp_test_function_2);
    handle.def("dict_check", &dict_check);
    handle.def("simulate_pu", &simulate_pu, "Simulates minimum bias PU event");
    handle.def("generate_pu_without_sim", &generate_pu_without_sim, "Generates PU particles and returns them but doesn't run simulation");
    handle.def("simulate_qqbar2ttbar", &simulate_qqbar2ttbar, "Simulates qqbar2ttbar PU event");
    handle.def("simulate_particle", &simulate_particle);
    handle.def("get_sensor_data", &get_sensor_data);
    handle.def("initialize", &initialize);
    handle.def("initialize_test", &initialize_test);
    handle.def("wrap_up", &wrap_up);
}