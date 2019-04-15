/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Options.hpp"

#include <iostream>

Options::Options() : desc("Allowed Options") {
    desc.add_options()
        ("help", "Prints this help message.")
        ("input,i", po::value<std::string>()->required(), "A Riegl Scan Project directory as input for the converter.")
        ("inputformat", po::value<std::string>()->default_value("rxp"), "The input pointcloud type in the riegl project folder to parse. Implemented: rxp, ascii")
        ("out,o", po::value<std::string>()->default_value("./scan_project"), "The directory where the converted scan project will be saved.")
        ("outformat", po::value<std::string>()->default_value("slam6d"), "The output coordinate space the converted scan project will be saved. Impemented: slam6d, lvr")
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
