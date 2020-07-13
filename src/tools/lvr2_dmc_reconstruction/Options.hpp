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
 * OptionsDMC.hpp
 *
 *  Created on: May 13, 2020
 *      Author: Benedikt SChumacher
 */
#ifndef OPTIONSDMC_H_
#define OPTIONSDMC_H_

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

namespace dmc_reconstruction
{

class Options : public BaseOption
{
  public:
    Options(int argc, char** argv);
    virtual ~Options();

    string getInputFileName() const;

    int getMaxLevel() const;

    float getMaxError() const;

    /*
     * prints information about needed command-line-inputs e.g: input-file (ply)
     */
    bool printUsage() const;

    int getKd() const;

    int getKn() const;

    int getKi() const;

    int getNumThreads() const;

    string getPCM() const;

    bool useRansac() const;

    string getScanPoseFile() const;


  private:
    /// The maximum allows octree level
    int m_maxLevel;

    /// The max allowed error between points and surfaces in an octree cell
    float m_maxError;

    /// The number of neighbors for distance function evaluation
    int m_kd;

    /// The number of neighbors for normal estimation
    int m_kn;

    /// The number of neighbors for normal interpolation
    int m_ki;

    /// The number of threads to use
    int m_numThreads;

    /// The used point cloud manager
    string m_pcm;
};

/// Output the Options - overloaded output Operator
inline ostream& operator<<(ostream& os, const Options& o)
{
    // o.printTransformation(os);

    cout << "##### InputFile-Name: " << o.getInputFileName() << endl;
    cout << "##### Max Level: " << o.getMaxLevel() << endl;
    cout << "##### Max Error: " << o.getMaxError() << endl;

    cout << "##### PCM: " << o.getPCM() << endl;
    cout << "##### KD: " << o.getKd() << endl;
    cout << "##### KI: " << o.getKi() << endl;
    cout << "##### KN: " << o.getKn() << endl;

    return os;
}
} // namespace dmc_reconstruction

#endif // OPTIONSDMC_H_
