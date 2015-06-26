/*
 * Main.cpp
 *
 *  Created on: 1.02.2013
 *      Author: Dominik Feldschnieders
 */


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

// Las vegas Toolkit
#include "io/PointBuffer.hpp"
#include "io/Model.hpp"
#include "io/ModelFactory.hpp"
#include "mpi/MPITree.hpp"
#include "geometry/ColorVertex.hpp"
#include "geometry/Normal.hpp"
#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "io/Progress.hpp"

#include <boost/program_options.hpp>
#include "largescale/LSTree.hpp"
#include <random>
#include <boost/iostreams/device/mapped_file.hpp>
#include <iostream>

using namespace lvr;

size_t countLines(string filename);

int main (int argc , char *argv[]) {
    // get all options

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(-500,500);
//    ofstream ofs("/home/isaak/meshing.largescale/dat/large.3d");
//    for(int i=0 ; i<500 ; i++)
//    {
//        for(int j=0 ; j<500 ; j++)
//        {
//            for(int k=0 ; k<500 ; k++)
//            {
//                ofs << distribution(generator) << " " << distribution(generator) << " " << distribution(generator) << "\n";
//            }
//        }
//    }

    boost::program_options::options_description desc("Allowed options");
    desc.add_options() ("file", boost::program_options::value<string>()->default_value("noinput"), "Inputfile");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    std::string filePath = vm["file"].as<string>();





    boost::iostreams::mapped_file_sink file;
    int numberOfElements = countLines(filePath) -1;
    int numberOfBytes = numberOfElements*sizeof(float)*3;

    boost::iostreams::mapped_file_params mfp;
    mfp.path = "asd.raw";
    mfp.new_file_size = numberOfElements*sizeof(float)*3;

    file.open(mfp);

    // Check if file was successfully opened
    if(file.is_open()) {
        // Get pointer to the data
        float * data = (float *)file.data();
        ifstream in;
        in.open(filePath.c_str());
        unsigned long long int bufferIndex = 0;
        std::cout << timestamp << "amount of Vertices: " << numberOfElements << std::endl;
        while (in.good() && bufferIndex < numberOfElements*3)
        {
            in >> data[bufferIndex] >> data[bufferIndex+1] >> data[bufferIndex+2];
            if(bufferIndex==0) std::cout << data[bufferIndex] << std::endl;
            bufferIndex+=3;
        }
        std::cout << timestamp << "last element: " << data[0] << "|"<< data[0+1]<<"|" << data[0+2] << " bufindex: " << bufferIndex << " numelem: " << numberOfElements << endl;
        file.close();

        LSTree tree(mfp.path, 100, 24000000, numberOfElements*3);
        std::cout << timestamp << "finished" << endl;

    } else {
        std::cout << timestamp << "could not map the file " << std::endl;
    }
}

size_t countLines(string filename)
{
    // Open file for reading
    ifstream in(filename.c_str());

    // Count lines in file
    size_t c = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        c++;
    }
    in.close();
    return c;
}

