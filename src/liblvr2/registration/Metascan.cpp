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
 * Metascan.cpp
 *
 *  @date Aug 1, 2019
 *  @author Malte Hillmann
 */
#include "lvr2/registration/Metascan.hpp"

namespace lvr2
{

Metascan::Metascan()
    : SLAMScanWrapper(ScanPtr(nullptr))
{

}

void Metascan::transform(const Transformd& transform, bool writeFrame, FrameUse use)
{
    for (auto& scan : m_scans)
    {
        scan->transform(transform, writeFrame, use);
    }
    m_deltaPose = transform * m_deltaPose;

    if (writeFrame)
    {
        addFrame(use);
    }
}

Vector3d Metascan::point(size_t index) const
{
    for (auto& scan : m_scans)
    {
        if (index < scan->numPoints())
        {
            return scan->point(index);
        }
        index -= scan->numPoints();
    }
    return Vector3d();
}

void Metascan::addScan(SLAMScanPtr scan)
{
    m_scans.push_back(scan);
    m_numPoints += scan->numPoints();
    m_deltaPose = scan->deltaPose();
}

} /* namespace lvr2 */
