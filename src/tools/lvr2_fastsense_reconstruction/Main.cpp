/*
 * Main.cpp
 *
 * Created on: 11.11.2020
 *      Author: Patrick Hoffmann (pahoffmann@uos.de)
 * 
 * Class used to reconstruct the GlobalMap from the FastsenseSLAM to inspect the qualit of the SLAM approach
 * 
 */

#include "lvr2/io/FastsenseIO.hpp"

#include <signal.h>

using namespace lvr2;


int main(int argc, char** argv)
{
    FastsenseIO fastsense;

    //todo: use options class for this, adapt return type to solution
    auto& map = fastsense.readMap("map.hdf5");

    //todo use the map data to create something that can be used by the marching cubes interface
    //create hashmap from data
    //bouding box important.
    //todo: force marc to do this as always

    return 0;
}
