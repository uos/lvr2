#include "Options.hpp"

#include <iostream>

Options::Options() : desc("Allowed Options") {
    desc.add_options()
        ("help", "Prints this help message.")
        ("input,i", po::value<std::string>()->required(), "A Riegl Scan Project directory as input for the converter.")
        ("out,o", po::value<std::string>()->default_value("./scan_project"), "The directory where the converted scan project will be saved.")
        ("reduction,r", po::value<unsigned int>()->default_value(1), "Reduces pointcloud size by importing only every Nth point (1 means no reduction).")
        ("start,s", po::value<unsigned int>()->default_value(1), "skipp the first start-1 scanpositions.")
        ("end,e", po::value<unsigned int>()->default_value(0), "skipp all scanpositions after end")
        ("force,f", "If this is set an already existing file will be overwritten in the output directory.")
    ;

    pod.add("input", 1); 

}

bool Options::parse(int argc, char **argv) {
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm); 
        po::notify(vm);
    }
    catch (po::error& e) {
        std::cout << "[Options] Error: " << e.what() << '\n' << desc <<std::endl;
        return false;
    }

    if (vm.count("help")) {
        std::cout << "[Options] Info: " << desc << std::endl;
        return false;
    }
    
    return true;
}
