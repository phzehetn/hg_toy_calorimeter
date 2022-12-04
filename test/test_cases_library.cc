//
// Created by Shah Rukh Qasim on 07.01.22.
//


#include "Calorimeter.hh"
#include "json/json.h"

int main(int argc, char*argv[]) {
//    simulate_pu(1, 0, "/Users/shahrukhqasim/Workspace/NextCal/ShahRukhStudies/scripts/toydetector/detector_specs.json");

    std::string s("{\"world_size_xy\": 4}");
    std::cout<<s<<std::endl;

    Json::Value root(s);
    Json::Reader reader;
    reader.parse(s, root);

    std::cout<<"Length of int "<<sizeof(int)<<std::endl;

    std::cout << root["world_size_xy"].asInt() <<std::endl;

//    dict_check(1, 0, "/Users/shahrukhqasim/Workspace/NextCal/ShahRukhStudies/scripts/toydetector/detector_specs.json");
}