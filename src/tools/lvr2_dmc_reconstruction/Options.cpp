/*
* OptionsDMC.cpp
*
*  Created on: May 13, 2020
*      Author: Benedikt Schumacher
*/

#include "Options.hpp"
#include "lvr2/config/lvropenmp.hpp"


using namespace boost::program_options;
namespace dmc_reconstruction{
    Options::Options(int argc, char **argv) : BaseOption(argc, argv) {
        m_descr.add_options()
                ("help", "Produce help message")
                ("inputFile",value<vector<string>>(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
                ("ml", value<int>(&m_maxLevel)->default_value(8), "Maximum allowed octree levels")
                ("e", value<float>(&m_maxError)->default_value(2.8), "Maximum allowed error between points and surfaces")
                ("kd", value<int>(&m_kd)->default_value(5), "Number of normals used for distance function evaluation")
                ("ki", value<int>(&m_ki)->default_value(10), "Number of normals used in the normal interpolation process")
                ("kn", value<int>(&m_kn)->default_value(10), "Size of k-neighborhood used for normal estimation")
                ("pcm,p", value<string>(&m_pcm)->default_value("FLANN"), "Point cloud manager used for point handling and normal estimation. Choose from {STANN, PCL, NABO}.")
                ("ransac", "Set this flag for RANSAC based normal estimation.")
                ("scanPoseFile", value<string>()->default_value(""), "ASCII file containing scan positions that can be used to flip normals")
                ("threads", value<int>(&m_numThreads)->default_value( lvr2::OpenMPConfig::getNumThreads() ), "Number of threads")
                ;
        setup();
    }

    Options::~Options() {
        //TODO: Auto-generated destructor stub
    }

    string Options::getInputFileName() const {
        return (m_variables["inputFile"].as<vector<string>>())[0];
    }

    int Options::getMaxLevel() const {
        return (m_variables["ml"].as<int>());
    }

    float Options::getMaxError() const {
        return (m_variables["e"].as<float>());
    }

    bool Options::printUsage() const {
        if (m_variables.count("help"))
        {
            cout << endl;
            cout << m_descr << endl;
            return true;
        }
        else if (!m_variables.count("inputFile"))
        {
            cout << "Error: You must specify an input file." << endl;
            cout << endl;
            cout << m_descr << endl;
            return true;
        }
        return false;
    }


    int Options::getKd() const {
        return m_variables["kd"].as<int>();
    }

    int Options::getKn() const {
        return m_variables["kn"].as<int>();
    }

    int Options::getKi() const {
        return m_variables["ki"].as<int>();
    }

    int Options::getNumThreads() const {
        return m_variables["threads"].as<int>();
    }

    string Options::getPCM() const {
        return (m_variables["pcm"].as<string>());
    }

    bool Options::useRansac() const
    {
        return (m_variables.count("ransac"));
    }

    string Options::getScanPoseFile() const
    {
        return (m_variables["scanPoseFile"].as<string>());
    }
}

