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
 * SlamAlign.cpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */
#include <lvr2/registration/SlamAlign.hpp>

using namespace std;

namespace lvr2
{

SlamAlign::SlamAlign(const std::vector<ScanPtr>& scans)
    : m_scans(move(scans))
{

}

void SlamAlign::match()
{
    string scan_number_string = to_string(m_scans.size());

    for(size_t i = 1; i < m_scans.size(); i++)
    {
        if (m_quiet)
        {
            cout << setw(scan_number_string.length()) << i << "/" << scan_number_string << ": " << flush;
        }
        else
        {
            cout << "Iteration " << setw(scan_number_string.length()) << i << "/" << scan_number_string << ": " << endl;
        }

        const ScanPtr& prev = m_scans[i - 1];
        const ScanPtr& cur = m_scans[i];

        ICPPointAlign icp(prev->getPoints(), cur->getPoints(), prev->getPose(), cur->getPose());
        icp.setMaxMatchDistance(m_icpMaxDistance);
        icp.setMaxIterations(m_icpIterations);
        icp.setEpsilon(m_epsilon);
        icp.setQuiet(m_quiet);

        Matrix4d result = icp.match();
        cur->transform(icp.getDeltaTransform());

        for(const ScanPtr& scan : m_scans)
        {
            if (scan != cur)
            {
                scan->addFrame();
            }
        }
    }
}

// ============================== Getters, Setters ==============================

void SlamAlign::setSlamMaxDistance(double slamMaxDistance)
{
    m_slamMaxDistance = slamMaxDistance;
}
void SlamAlign::setSlamIterations(int slamIterations)
{
    m_slamIterations = slamIterations;
}
void SlamAlign::setIcpMaxDistance(double icpMaxDistance)
{
    m_icpMaxDistance = icpMaxDistance;
}
void SlamAlign::setIcpIterations(int icpIterations)
{
    m_icpIterations = icpIterations;
}
void SlamAlign::setDoLoopClosing(bool doLoopClosing)
{
    m_doLoopClosing = doLoopClosing;
}
void SlamAlign::setDoGraphSlam(bool doGraphSlam)
{
    m_doGraphSlam = doGraphSlam;
}
void SlamAlign::setEpsilon(double epsilon)
{
    m_epsilon = epsilon;
}
void SlamAlign::setQuiet(bool quiet)
{
    m_quiet = quiet;
}

double SlamAlign::getSlamMaxDistance() const
{
    return m_slamMaxDistance;
}
int SlamAlign::getSlamIterations() const
{
    return m_slamIterations;
}
double SlamAlign::getIcpMaxDistance() const
{
    return m_icpMaxDistance;
}
int SlamAlign::getIcpIterations() const
{
    return m_icpIterations;
}
bool SlamAlign::getDoLoopClosing() const
{
    return m_doLoopClosing;
}
bool SlamAlign::getDoGraphSlam() const
{
    return m_doGraphSlam;
}
double SlamAlign::getEpsilon() const
{
    return m_epsilon;
}
bool SlamAlign::getQuiet() const
{
    return m_quiet;
}

} /* namespace lvr2 */
