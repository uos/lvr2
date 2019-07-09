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
#include <lvr2/registration/SlamOptions.hpp>

namespace lvr2
{

class SlamAlign
{

public:
    SlamAlign(const SlamOptions& options = SlamOptions(), int count = -1);

    void addScan(const ScanPtr& scan);
    void setScan(int index, const ScanPtr& scan);

    void match();

    void writeFrames(int scan, std::string file)
    {
        m_scans[scan]->writeFrames(file);
    }

    virtual ~SlamAlign() = default;

    void setOptions(const SlamOptions& options)
    {
        m_options = options;
    }
    SlamOptions& options()
    {
        return m_options;
    }

protected:

    void reduceScan(const ScanPtr& scan);

    void applyTransform(ScanPtr scan, const Matrix4f& transform);
    void addFrame(ScanPtr current);
    void checkLoopClose(int last);
    void loopClose(int first, int last);
    void graphSlam(int last);
    void findCloseScans(int scan, vector<int>& output);
    Matrix6f eulerCovariance(int a, int b) const;

    SlamOptions             m_options;

    vector<ScanPtr>         m_scans;

    ScanPtr                 m_metascan;

    vector<pair<int, int>>  m_graph;
    bool                    m_foundLoop;
};

} /* namespace lvr2 */

#endif /* SLAMALIGN_HPP_ */
