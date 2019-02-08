//
// Created by patrick on 1/30/19.
//

#include "OptionsGSS.hpp"
using namespace gs_reconstruction;


Options::Options(int argc, char **argv) : BaseOption(argc, argv) {
    m_descr.add_options()
            ("help", "Produce help message")
            ("inputFile",value<vector<string>>()->required(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
            ("runtime", value<int>(&m_runtime)->default_value(3000), "Overall runtime of the Algorithm. The longer, the (better), default: 3000")
            ("basicSteps",value<int>(&m_basicSteps)->default_value(10), "Basics step until split, default: 10")
            ("numSplits",value<int>(&m_numSplits)->default_value(5), "splits per runtime, default: 5")
            ("boxFactor",value<float>(&m_boxFactor)->default_value(1), "scale of initial mesh, default: 1")
            ("withCollapse",value<bool>(&m_withCollapse)->default_value(true), "with edge collapse? default: true")
            ("learningRate",value<float>(&m_learningRate)->default_value(0.1),"learning rate of winner vertex, default: 1")
            ("neighborLearningRate",value<float>(&m_neighborLearningRate)->default_value(0.08),"learning rate of winner vertex neighbours, default: 0.08")
            ("decreaseFactor",value<float>(&m_decreaseFactor)->default_value(1.0),"dynamic decrease factor, default start: 1.0")
            ("allowMiss",value<int>(&m_allowMiss)->default_value(7), "allow miss of vertex, default: 7")
            ("collapseThreshold",value<float>(&m_collapseThreshold)->default_value(0.3), "threshold for collapse, default: 0.3")
            ("filterChain",value<bool>(&m_filterChain)->default_value(false),"should the filter chain run? default: false")
            ("deleteLongEdgesFactor",value<int>(&m_deleteLongEdgesFactor)->default_value(10), "0 = no deleting, default: 10")
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
    return m_variables["m_runtime"].as<int>();
}

int Options::getBasicSteps() const {
    return m_variables["m_basicSteps"].as<int>();
}

int Options::getNumSplits() const {
    return m_variables["m_numSplits"].as<int>();
}

float Options::getBoxFactor() const {
    return m_variables["m_boxFactor"].as<float>();
}

bool Options::getWithCollapse() const {
    return m_variables["m_withCollapse"].as<bool>();
}

float Options::getLearningRate() const {
    return m_variables["m_learningRate"].as<float>();
}

float Options::getNeighborLearningRate() const {
    return m_variables["m_neighborLearningRate"].as<float>();
}

float Options::getDecreaseFactor() const {
    return m_variables["m_decreaseFactor"].as<float>();

}

int Options::getAllowMiss() const {
    return m_variables["m_allowMiss"].as<int>();
}

float Options::getCollapseThreshold() const {
    return m_variables["m_collapseThreshold"].as<float>();
}

bool Options::isFilterChain() const {
    //return m_variables["m_basicSteps"].as<bool>();
}

int Options::getDeleteLongEdgesFactor() const {
    return m_variables["m_deleteLongEdgesFactor"].as<int>();
}


bool Options::printUsage() const {
    if(!m_variables.count("inputFile"))
    {
        cout << "Error: You must specify an input file." << endl;
        cout << endl;
        cout << m_descr << endl;
        return true;
    }

    if(m_variables.count("help"))
    {
        cout << endl;
        cout << m_descr << endl;
        return true;
    }
    return false;
}
