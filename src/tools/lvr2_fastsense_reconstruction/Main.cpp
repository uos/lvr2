/**
 * Main.cpp
 *
 * Created on: 11.11.2020
 * Author: Steffen Hinderink, Marc Eisoldt, Juri Vana, (Patrick Hoffmann)
 * 
 * Function used to reconstruct a mesh from the GlobalMap of the FastSenseSLAM to inspect the quality of the SLAM approach
 */

#include <iostream>
#include <highfive/H5File.hpp>

int main(int argc, char** argv)
{
    /*
    1. Einlesen
    2. In geerbte Klasse von HashGrid einfügen
    3. Mit HashGrid FastReconstruction aufrufen
    4. Mesh mit vorhandenen IO rausschreiben
    */

    // 1
    HighFive::File f("/home/fastsense/Develop/map.h5", HighFive::File::ReadOnly);
    //HighFive::Group g = f.getGroup("/map");
    //HighFive::DataSet d = g.getDataSet("-1_0_0"); 

    std::cout << "Letzter Test für heute :D" << std::endl;

    return 0;
}
