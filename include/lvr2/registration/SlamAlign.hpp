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
 * SlamAlign.hpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */
#ifndef SLAMALIGN_HPP_
#define SLAMALIGN_HPP_

#include <lvr2/registration/ICPPointAlign.hpp>
#include <lvr2/registration/Scan.hpp>

namespace lvr2
{

class SlamAlign
{

public:
    SlamAlign(const std::vector<ScanPtr>& scans);

    void match();

    virtual ~SlamAlign() = default;

    void    setDoLoopClosing(bool doLoopClosing);
    void    setDoGraphSlam(bool doGraphSlam);
    void    setSlamMaxDistance(double slamMaxDistance);
    void    setSlamIterations(int slamIterations);

    void    setIcpMaxDistance(double icpMaxDistance);
    void    setIcpIterations(int icpIterations);

    void    setMinDistance(double minDistance);
    void    setMaxDistance(double maxDistance);
    void    setReduction(double reduction);

    void    setTrustPose(bool trustPose);
    void    setMetascan(bool metascan);

    void    setEpsilon(double epsilon);
    void    setQuiet(bool quiet);


    bool    getDoLoopClosing() const;
    bool    getDoGraphSlam() const;
    double  getSlamMaxDistance() const;
    int     getSlamIterations() const;
    double  getIcpMaxDistance() const;
    int     getIcpIterations() const;
    double  getMinDistance() const;
    double  getMaxDistance() const;
    double  getReduction() const;
    bool    getTrustPose() const;
    bool    getMetascan() const;
    double  getEpsilon() const;
    bool    getQuiet() const;

protected:

    void applyTransform(ScanPtr scan, const Matrix4d& transform);

    bool    m_doLoopClosing = false;
    bool    m_doGraphSlam = false;
    double  m_slamMaxDistance = 25;
    int     m_slamIterations = 50;

    double  m_icpMaxDistance = 25;
    int     m_icpIterations = 50;

    double  m_minDistance = -1;
    double  m_maxDistance = -1;
    double  m_reduction = -1;

    bool    m_trustPose = false;
    bool    m_metascan = false;

    double  m_epsilon = 0.00001;
    bool    m_quiet = false;

    vector<ScanPtr> m_scans;
};

} /* namespace lvr2 */

#endif /* SLAMALIGN_HPP_ */
