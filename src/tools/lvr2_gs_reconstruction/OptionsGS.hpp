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

/*
 * OptionsGSS.hpp
 *
 *  Created on: Feb 09, 2019
 *      Author: Patrick Hoffmann
 */
#ifndef OPTIONSGS_H_
#define OPTIONSGS_H_

#include "lvr2/config/BaseOption.hpp"

#include <boost/program_options.hpp>
#include <float.h>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::ostream;
using std::string;
using std::vector;
using namespace lvr2;

namespace gs_reconstruction
{

class Options : public BaseOption
{
  public:
    Options(int argc, char** argv);
    virtual ~Options();

    int getRuntime() const;

    int getBasicSteps() const;

    int getNumSplits() const;

    float getBoxFactor() const;

    bool getWithCollapse() const;

    float getLearningRate() const;

    float getNeighborLearningRate() const;

    float getDecreaseFactor() const;

    int getAllowMiss() const;

    float getCollapseThreshold() const;

    bool isFilterChain() const;

    int getDeleteLongEdgesFactor() const;

    bool isInterior() const;

    int getNumBalances() const;

    string getInputFileName() const;

    /*
     * prints information about needed command-line-inputs e.g: input-file (ply)
     */
    bool printUsage() const;

    int getKd() const;

    int getKn() const;

    int getKi() const;

    string getPcm() const;

  private:
    int m_runtime;
    int m_basicSteps;
    int m_numSplits;
    float m_boxFactor;
    bool m_withCollapse;
    float m_learningRate;
    float m_neighborLearningRate;
    float m_decreaseFactor;
    int m_allowMiss;
    float m_collapseThreshold;
    bool m_filterChain;
    int m_deleteLongEdgesFactor;
    bool m_interior;
    int m_balances;
    /// The number of neighbors for distance function evaluation
    int m_kd;

    /// The number of neighbors for normal estimation
    int m_kn;

    /// The number of neighbors for normal interpolation
    int m_ki;

    /// The used point cloud manager
    string m_pcm;
};

/// Output the Options - overloaded output Operator
inline std::ostream& operator<<(std::ostream& os, const Options& o)
{
    // o.printTransformation(os);

    std::cout << "##### InputFile-Name: " << o.getInputFileName() << std::endl;
    std::cout << "##### Runtime: " << o.getRuntime() << std::endl;
    std::cout << "##### BasicSteps: " << o.getBasicSteps() << std::endl;
    std::cout << "##### NumSplits: " << o.getNumSplits() << std::endl;
    std::cout << "##### BoxFactor: " << o.getBoxFactor() << std::endl;
    std::cout << "#### WithCollapse: " << o.getWithCollapse() << std::endl;
    std::cout << "##### LearningRate: " << o.getLearningRate() << std::endl;
    std::cout << "##### NeighbourLearningRate: " << o.getNeighborLearningRate() << std::endl;
    std::cout << "##### DecreaseFactor: " << o.getDecreaseFactor() << std::endl;
    std::cout << "##### AllowMiss: " << o.getAllowMiss() << std::endl;
    std::cout << "##### CollapseThreshold: " << o.getCollapseThreshold() << std::endl;
    std::cout << "##### FilterChain: " << o.isFilterChain() << std::endl;
    std::cout << "##### DeleteLongEdgesFactor: " << o.getDeleteLongEdgesFactor() << std::endl;
    std::cout << "##### Interior: " << o.isInterior() << std::endl;
    std::cout << "##### Balances: " << o.getNumBalances() << std::endl;
    std::cout << "##### PCM: " << o.getPcm() << std::endl;
    std::cout << "##### KD: " << o.getKd() << std::endl;
    std::cout << "##### KI: " << o.getKi() << std::endl;
    std::cout << "##### KN: " << o.getKn() << std::endl;

    return os;
}
} // namespace gs_reconstruction

#endif // OPTIONSGS_H_
