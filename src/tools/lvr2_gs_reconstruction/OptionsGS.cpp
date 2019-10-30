/*
* OptionsGSS.cpp
*
*  Created on: Feb 09, 2019
*      Author: Patrick Hoffmann
*/

#include "OptionsGS.hpp"


using namespace boost::program_options;
namespace gs_reconstruction{
    Options::Options(int argc, char **argv) : BaseOption(argc, argv) {
        m_descr.add_options()
                ("help", "Produce help message")
                ("inputFile",value<vector<string>>(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
                ("runtime", value<int>(&m_runtime)->default_value(3000), "Overall runtime of the Algorithm. The longer, the (better), default: 3000")
                ("basicSteps",value<int>(&m_basicSteps)->default_value(10), "Basics step until split, default: 10")
                ("numSplits",value<int>(&m_numSplits)->default_value(5), "splits per runtime, default: 5")
                ("boxFactor",value<float>(&m_boxFactor)->default_value(0.2), "scale of initial mesh, default: 1")
                ("withCollapse",value<bool>(&m_withCollapse)->default_value(true), "with edge collapse? default: true")
                ("learningRate",value<float>(&m_learningRate)->default_value(0.1),"learning rate of winner vertex, default: 1")
                ("neighborLearningRate",value<float>(&m_neighborLearningRate)->default_value(0.08),"learning rate of winner vertex neighbours, default: 0.08")
                ("decreaseFactor",value<float>(&m_decreaseFactor)->default_value(0.999),"dynamic decrease factor, default start: 1.0")
                ("allowMiss",value<int>(&m_allowMiss)->default_value(7), "allow miss of vertex, default: 7")
                ("collapseThreshold",value<float>(&m_collapseThreshold)->default_value(0.3), "threshold for collapse, default: 0.3")
                ("filterChain",value<bool>(&m_filterChain)->default_value(false),"should the filter chain run? default: false")
                ("deleteLongEdgesFactor",value<int>(&m_deleteLongEdgesFactor)->default_value(10), "0 = no deleting, default: 10")
                ("interior",value<bool>(&m_interior)->default_value(false), "false: reconstruct exterior, true: reconstruct interior")
                ("balances",value<int>(&m_balances)->default_value(20), "Number of TumbleTree-Balances during the reconstruction. default: 20")
                ("kd", value<int>(&m_kd)->default_value(5), "Number of normals used for distance function evaluation")
                ("ki", value<int>(&m_ki)->default_value(10), "Number of normals used in the normal interpolation process")
                ("kn", value<int>(&m_kn)->default_value(10), "Size of k-neighborhood used for normal estimation")
                ("pcm,p", value<string>(&m_pcm)->default_value("FLANN"), "Point cloud manager used for point handling and normal estimation. Choose from {STANN, PCL, NABO}.")
                ;
        setup();
    }

    Options::~Options() {
        //TODO: Auto-generated destructor stub
    }

    string Options::getInputFileName() const {
        return (m_variables["inputFile"].as<vector<string>>())[0];
    }

    int Options::getRuntime() const {
        return m_variables["runtime"].as<int>();
    }

    int Options::getBasicSteps() const {
        return m_variables["basicSteps"].as<int>();
    }

    int Options::getNumSplits() const {
        return m_variables["numSplits"].as<int>();
    }

    float Options::getBoxFactor() const {
        return m_variables["boxFactor"].as<float>();
    }

    bool Options::getWithCollapse() const {
        return m_variables["withCollapse"].as<bool>();
    }

    float Options::getLearningRate() const {
        return m_variables["learningRate"].as<float>();
    }

    float Options::getNeighborLearningRate() const {
        return m_variables["neighborLearningRate"].as<float>();
    }

    float Options::getDecreaseFactor() const {
        return m_variables["decreaseFactor"].as<float>();

    }
    int Options::getAllowMiss() const {
        return m_variables["allowMiss"].as<int>();
    }

    float Options::getCollapseThreshold() const {
        return m_variables["collapseThreshold"].as<float>();
    }

    bool Options::isFilterChain() const {
        return m_variables["filterChain"].as<bool>();
    }

    int Options::getDeleteLongEdgesFactor() const {
        return m_variables["deleteLongEdgesFactor"].as<int>();
    }

    bool Options::isInterior() const {
        return m_variables["interior"].as<bool>();
    }

    int Options::getNumBalances() const {
        return m_variables["balances"].as<int>();
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

    string Options::getPcm() const {
        return (m_variables["pcm"].as<string>());
    }
}

