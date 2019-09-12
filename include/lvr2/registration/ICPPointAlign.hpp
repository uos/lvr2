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

/**
 * ICPPointAlign.hpp
 *
 *  @date Mar 18, 2014
 *  @author Thomas Wiemann
 */
#ifndef ICPPOINTALIGN_HPP_
#define ICPPOINTALIGN_HPP_

#include "KDTree.hpp"
#include "SLAMScanWrapper.hpp"

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

/**
 * @brief A class to align two Scans with ICP
 * 
 */
class ICPPointAlign
{
public:
    /**
     * @brief Construct a new ICPPointAlign object. Data is transformed to match Model
     * 
     * @param model The Model Scan (stays unchanged)
     * @param data The Data Scan (transformed)
     */
    ICPPointAlign(SLAMScanPtr model, SLAMScanPtr data);

    /**
     * @brief Executes the ICPAlign
     * 
     * @return Transformd The delta transformation caused by this Method
     */
    Transformd match();

    virtual ~ICPPointAlign() = default;

    void    setMaxMatchDistance(double distance);
    void    setMaxIterations(int iterations);
    void    setMaxLeafSize(int maxLeafSize);
    void    setEpsilon(double epsilon);
    void    setVerbose(bool verbose);

    double  getMaxMatchDistance() const;
    int     getMaxIterations() const;
    int     getMaxLeafSize() const;
    double  getEpsilon() const;
    bool    getVerbose() const;

protected:

    double      m_epsilon;
    double      m_maxDistanceMatch;
    int         m_maxIterations;
    int         m_maxLeafSize;

    bool        m_verbose;

    SLAMScanPtr m_modelCloud;
    SLAMScanPtr m_dataCloud;

    KDTreePtr   m_searchTree;
};

} /* namespace lvr2 */

#endif /* ICPPOINTALIGN_HPP_ */
