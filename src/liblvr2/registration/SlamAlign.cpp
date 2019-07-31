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

#include <iomanip>

using namespace std;

namespace lvr2
{

SlamAlign::SlamAlign(const SlamOptions& options, vector<ScanPtr>& scans)
    : m_options(options), m_scans(scans), m_graph(&m_options), m_foundLoop(false)
{
    // The first Scan is never changed
    m_alreadyMatched = 1;

    for (auto& scan : m_scans)
    {
        reduceScan(scan);
    }
}

SlamAlign::SlamAlign(const SlamOptions& options)
    : m_options(options), m_graph(&m_options), m_foundLoop(false)
{
    // The first Scan is never changed
    m_alreadyMatched = 1;
}

void SlamAlign::setOptions(const SlamOptions& options)
{
    m_options = options;
}

SlamOptions& SlamAlign::options()
{
    return m_options;
}

void SlamAlign::addScan(const ScanPtr& scan, bool match)
{
    reduceScan(scan);
    m_scans.push_back(scan);

    if (match)
    {
        this->match();
    }
}

ScanPtr SlamAlign::getScan(size_t index)
{
    return m_scans[index];
}

void SlamAlign::reduceScan(const ScanPtr& scan)
{
    size_t prev = scan->count();
    if (m_options.reduction >= 0)
    {
        scan->reduce(m_options.reduction);
    }
    if (m_options.minDistance >= 0)
    {
        scan->setMinDistance(m_options.minDistance);
    }
    if (m_options.maxDistance >= 0)
    {
        scan->setMaxDistance(m_options.maxDistance);
    }

    if (scan->count() < prev && m_options.verbose)
    {
        cout << "Removed " << (prev - scan->count()) << " / " << prev << " Points -> " << scan->count() << " left" << endl;
    }
}

void SlamAlign::match()
{
    // need at least 2 Scans
    if (m_scans.size() <= 1)
    {
        return;
    }

    if (m_options.metascan && !m_metascan)
    {
        createMetascan();
    }

    string scan_number_string = to_string(m_scans.size() - 1);

    // only match everything after m_alreadyMatched
    for (; m_alreadyMatched < m_scans.size(); m_alreadyMatched++)
    {
        size_t i = m_alreadyMatched;

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

        if (!m_options.trustPose && i != 1) // no deltaPose on first run
        {
            applyTransform(cur, prev->getDeltaPose());
        }
        else
        {
            applyTransform(cur, Matrix4d::Identity());
        }

        ICPPointAlign icp(prev, cur);
        icp.setMaxMatchDistance(m_options.icpMaxDistance);
        icp.setMaxIterations(m_options.icpIterations);
        icp.setMaxLeafSize(m_options.maxLeafSize);
        icp.setEpsilon(m_options.epsilon);
        icp.setVerbose(m_options.verbose);

        icp.match();

        applyTransform(cur, Matrix4d::Identity());

        if (m_options.metascan)
        {
            m_metascan->addScanToMeta(cur);
        }

        checkLoopClose(i);
    }
}

void SlamAlign::applyTransform(ScanPtr scan, const Matrix4d& transform)
{
    scan->transform(transform);

    bool found = false;
    for (const ScanPtr& s : m_scans)
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

void SlamAlign::createMetascan()
{
    m_metascan = ScanPtr(new Scan(*m_scans[0]));

    for (size_t i = 1; i < m_alreadyMatched; i++)
    {
        m_metascan->addScanToMeta(m_scans[i]);
    }
}

void SlamAlign::checkLoopClose(size_t last)
{
    if (!m_options.doLoopClosing && !m_options.doGraphSlam)
    {
        return;
    }

    m_graph.addEdge(last - 1, last);

    bool hasLoop = false;

    size_t first = 0;

    vector<size_t> others;
    if (findCloseScans(last, others))
    {
        hasLoop = true;
        first = others[0];

        for (size_t i : others)
        {
            m_graph.addEdge(i, last);
        }
    }

    if (hasLoop && m_options.doLoopClosing)
    {
        loopClose(first, last);
        if (m_options.metascan)
        {
            createMetascan();
        }
    }

    // wait for falling edge
    if (m_foundLoop && !hasLoop && m_options.doGraphSlam)
    {
        graphSlam(last);
        if (m_options.metascan)
        {
            createMetascan();
        }
    }

    m_foundLoop = hasLoop;
}

void SlamAlign::loopClose(size_t first, size_t last)
{
    cout << "Loopclose " << first << " -> " << last << endl;

    ScanPtr metaFirst = make_shared<Scan>(*m_scans[first]);
    metaFirst->addScanToMeta(m_scans[first + 1]);
    metaFirst->addScanToMeta(m_scans[first + 2]);

    ScanPtr metaLast = make_shared<Scan>(*m_scans[last]);
    metaLast->addScanToMeta(m_scans[last - 1]);
    metaLast->addScanToMeta(m_scans[last - 2]);

    ICPPointAlign icp(metaFirst, metaLast);
    icp.setMaxMatchDistance(m_options.slamMaxDistance);
    icp.setMaxIterations(m_options.slamIterations);
    icp.setMaxLeafSize(m_options.maxLeafSize);
    icp.setEpsilon(m_options.epsilon);
    icp.setVerbose(m_options.verbose);

    Matrix4d transform = icp.match();

    cout << "Loopclose delta: " << endl << transform << endl << endl;

    for (size_t i = first; i <= last; i++)
    {
        float factor = (i - first) / (float)(last - first);

        Matrix4d delta = (transform - Matrix4d::Identity()) * factor + Matrix4d::Identity();

        m_scans[i]->transform(delta, true, ScanUse::LOOPCLOSE);
    }

    // Add frame to unaffected scans
    for (size_t i = 0; i < first; i++)
    {
        m_scans[i]->addFrame();
    }
    for (size_t i = last + 1; i < m_scans.size(); i++)
    {
        m_scans[i]->addFrame(ScanUse::INVALID);
    }
}

void SlamAlign::graphSlam(size_t last)
{
    m_graph.doGraphSlam(m_scans, last);
}

bool SlamAlign::findCloseScans(size_t scan, vector<size_t>& output)
{
    ScanPtr& cur = m_scans[scan];

    // closeLoopPairs not specified => use closeLoopDistance
    if (m_options.closeLoopPairs < 0)
    {
        float maxDist = std::pow(m_options.closeLoopDistance, 2);
        Vector3f pos = cur->getPosition();
        for (size_t other = 0; other < scan - m_options.loopSize; other++)
        {
            float dist = (m_scans[other]->getPosition() - pos).squaredNorm();
            if (dist < maxDist)
            {
                output.push_back(other);
            }
        }
    }
    else
    {
        // convert current Scan to KDTree for Pair search
        auto tree = KDTree::create(cur->points(), cur->count(), m_options.maxLeafSize);

        size_t maxLen = 0;
        for (size_t other = 0; other < scan - m_options.loopSize; other++)
        {
            maxLen = max(maxLen, m_scans[other]->count());
        }

        Vector3f** neighbors = new Vector3f*[maxLen];

        for (size_t other = 0; other < scan - m_options.loopSize; other++)
        {
            const ScanPtr& scan = m_scans[other];
            size_t count = getNearestNeighbors(tree, scan->points(), neighbors, scan->count(), m_options.slamMaxDistance);

            if (count >= m_options.closeLoopPairs)
            {
                output.push_back(other);
            }
        }

        delete[] neighbors;
    }

    return !output.empty();
}

} /* namespace lvr2 */
