/*
 * MainGS.cpp
 *
 *  Created on: somewhen.02.2019
 *      Author: Patrick Hoffmann (pahoffmann@uos.de)
 */

/// New includes, to be evaluated, which we actually need

#include "lvr2/io/FastsenseIO.hpp"

#include <signal.h>

using namespace lvr2;


int main(int argc, char** argv)
{
    FastsenseIO fastsense;
    fastsense.readMap("map.hdf5");

    return 0;
}
