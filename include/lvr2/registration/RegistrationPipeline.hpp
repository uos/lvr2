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

#ifndef REGISTRATION_PIPELINE_OBJECT_H_
#define REGISTRATION_PIPELINE_OBJECT_H_

#include "lvr2/registration/SLAMOptions.hpp"
#include "lvr2/registration/GraphSLAM.hpp"
#include "lvr2/types/ScanTypes.hpp"
/**
 * RegistrationPipeline.hpp
 *
 *  @date Jan 7, 2020
 *  @author Timo Osterkamp (tosterkamp@uni-osnabrueck.de)
 *  @author Wilko Müller
 */


using namespace lvr2;

class RegistrationPipeline
{
public:

    /**
     * @brief Construct a new RegistrationPipeline object.
     * 
     * @param options The SLAM Options 
     * @param scans The scan project
     */
    RegistrationPipeline(const SLAMOptions* options, ScanProjectEditMarkPtr scans);

    /**
     * @brief Starts the registration
     * 
     * Starts the registration. Uses the SLAMOptions given in the constructor.
     * */
    void doRegistration();
private:

    /**
     * @brief Metric to determine wether the given matrices are too different from each other
     * 
     * @param a Transformation matrix
     * @param b Transformation matrix
     * @return true if a and b are in tolerated range
     * @return false if difference between a and b is too big
     */
    bool isToleratedDifference(Transformd a, Transformd b);

    /**
     * @brief Rotates the given 4x4 matrix around the y-axis
     * @param inputMatrix4x4 The matrix getting transformed
     * @param angle The rotation angle in degree
     * 
     * Rotates the given 4x4 matrix around the y-axis. For the Situation, where the scanner
     * was mounted at an incorrect angle. Can be used when all scans have the same angle offset.
     * */
    void rotateAroundYAxis(Transformd *inputMatrix4x4, double angle);

    const SLAMOptions* m_options;
    ScanProjectEditMarkPtr m_scans;
};

#endif // REGISTRATION_PIPELINE_OBJECT_H_