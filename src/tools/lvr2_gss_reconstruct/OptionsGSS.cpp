//
// Created by patrick on 1/30/19.
//

#include "OptionsGSS.hpp"
using namespace gss_reconstruct;
int Options::getRuntime() const {
    return m_runtime;
}

int Options::getBasicSteps() const {
    return m_basicSteps;
}

int Options::getNumSplits() const {
    return m_numSplits;
}

float Options::getBoxFactor() const {
    return m_boxFactor;
}

int Options::getWithCollapse() const {
    return m_withCollapse;
}

float Options::getLearningRate() const {
    return m_learningRate;
}

float Options::getNeighborLearningRate() const {
    return m_neighborLearningRate;
}

float Options::getDecreaseFactor() const {
    return m_decreaseFactor;
}

int Options::getAllowMiss() const {
    return m_allowMiss;
}

float Options::getCollapseThreshold() const {
    return m_collapseThreshold;
}

bool Options::isFilterChain() const {
    return m_filterChain;
}

int Options::getDeleteLongEdgesFactor() const {
    return m_deleteLongEdgesFactor;
}

string Options::getInputFileName() const {
    return m_variables["inputFileName"].as<string>();
}
