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
*  Created on: Nov 21, 2010
*      Author: Patrick Hoffmann
*/
#ifndef LAS_VEGAS_OPTIONSGSS_H
#define LAS_VEGAS_OPTIONSGSS_H

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <float.h>

#include <lvr2/config/BaseOption.hpp>

using std::ostream;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace lvr2;

namespace gss_reconstruct{

    class Options : public BaseOption {
    public:
        Options(int argc, char** argv);
        virtual ~Options();

        int getRuntime() const;

        int getBasicSteps() const;

        int getNumSplits() const;

        float getBoxFactor() const;

        int getWithCollapse() const;

        float getLearningRate() const;

        float getNeighborLearningRate() const;

        float getDecreaseFactor() const;

        int getAllowMiss() const;

        float getCollapseThreshold() const;

        bool isFilterChain() const;

        int getDeleteLongEdgesFactor() const;

        string getInputFileName() const;

    private:
        int m_runtime;
        int m_basicSteps;
        int m_numSplits;
        float m_boxFactor;
        int m_withCollapse;
        float m_learningRate;
        float m_neighborLearningRate;
        float m_decreaseFactor;
        int m_allowMiss;
        float m_collapseThreshold;
        bool m_filterChain;
        int m_deleteLongEdgesFactor;
        string m_inputFileName;

    };
}



#endif //LAS_VEGAS_OPTIONSGSS_H
