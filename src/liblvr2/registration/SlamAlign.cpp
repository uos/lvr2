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
#include <lvr2/registration/MetaScan.hpp>

#include <iomanip>

using namespace std;

namespace lvr2
{

SlamAlign::SlamAlign(const std::vector<ScanPtr>& scans, const SlamOptions& options)
    : m_scans(move(scans)), m_options(options)
{

}

void SlamAlign::match()
{
    if (m_options.reduction >= 0 || m_options.minDistance >= 0 || m_options.maxDistance >= 0)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < m_scans.size(); i++)
        {
            if (m_options.reduction >= 0)
            {
                m_scans[i]->reduce(m_options.reduction);
            }
            if (m_options.minDistance >= 0)
            {
                m_scans[i]->setMinDistance(m_options.minDistance);
            }
            if (m_options.maxDistance >= 0)
            {
                m_scans[i]->setMaxDistance(m_options.maxDistance);
            }
        }
    }

    if (m_options.metascan)
    {
        MetaScan* meta = new MetaScan();
        meta->addScan(m_scans[0]);
        m_metascan = ScanPtr(meta);
    }

    string scan_number_string = to_string(m_scans.size() - 1);
    for (int i = 1; i < m_scans.size(); i++)
    {
        if (m_options.verbose)
        {
            cout << "Iteration " << setw(scan_number_string.length()) << i << "/" << scan_number_string << ": " << endl;
        }
        else
        {
            cout << setw(scan_number_string.length()) << i << "/" << scan_number_string << ": " << flush;
        }

        ScanPtr prev = m_options.metascan ? m_metascan : m_scans[i - 1];
        const ScanPtr& cur = m_scans[i];

        if (!m_options.trustPose)
        {
            applyTransform(cur, prev->getDeltaPose());
        }
        else
        {
            // create frame entry
            applyTransform(cur, Matrix4d::Identity());
        }

        ICPPointAlign icp(prev, cur);
        icp.setMaxMatchDistance(m_options.icpMaxDistance);
        icp.setMaxIterations(m_options.icpIterations);
        icp.setEpsilon(m_options.epsilon);
        icp.setVerbose(m_options.verbose);

        Matrix4d result = icp.match();

        applyTransform(cur, icp.getDeltaTransform());

        if (m_options.metascan && i + 1 < m_scans.size())
        {
            MetaScan* meta = (MetaScan*)(m_metascan.get());
            meta->addScan(cur);
        }

        checkLoopClose(i);
    }
}

void SlamAlign::applyTransform(ScanPtr scan, const Matrix4d& transform)
{
    scan->transform(transform);

    bool found = false;
    for(const ScanPtr& s : m_scans)
    {
        if (s != scan)
        {
            s->addFrame(found ? ScanUse::INVALID : ScanUse::UNUSED);
        }
        else
        {
            found = true;
        }
    }
}

void SlamAlign::checkLoopClose(int last)
{
    if (!m_options.doLoopClosing && !m_options.doGraphSlam)
    {
        return;
    }

    const ScanPtr& cur = m_scans[last];

    int first = -1;

    // no Pairs => use closeLoopDistance
    if (m_options.closeLoopPairs < 0)
    {
        double maxDist = std::pow(m_options.closeLoopDistance, 2);
        Vector3d pos = cur->getPosition();
        for (int other = 0; other < last; other++)
        {
            double dist = (m_scans[other]->getPosition() - pos).squaredNorm();
            if (dist < maxDist)
            {
                first = other;
                break;
            }
        }
    }
    else
    {
        // convert current Scan to KDTree for Pair search
        size_t n = cur->count();
        PointArray points = PointArray(new Vector3d[n]);

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++)
        {
            points[i] = cur->getPointTransformed(i);
        }
        auto tree = KDTree::create(points, n);

        for (int other = 0; other < last; other++)
        {
            const ScanPtr& scan = m_scans[other];
            int count = 0;

            #pragma omp parallel for reduction(+:count)
            for (size_t i = 0; i < scan->count(); i++)
            {
                Vector3d point = scan->getPointTransformed(i);
                Vector3d* neighbor = nullptr;
                double dist;
                tree->nearestNeighbor(point, neighbor, dist, m_options.slamMaxDistance);
                if (neighbor != nullptr)
                {
                    count++;
                }
            }

            if (count >= m_options.closeLoopPairs)
            {
                first = other;
                break;
            }
        }
    }

    if (first != -1)
    {
        closeLoop(first, last);
    }
}

void SlamAlign::closeLoop(int first, int last)
{
    // TODO: implement
}

} /* namespace lvr2 */
