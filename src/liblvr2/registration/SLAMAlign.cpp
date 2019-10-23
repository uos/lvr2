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
 * SLAMAlign.cpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */

#include "lvr2/registration/SLAMAlign.hpp"
#include "lvr2/registration/ICPPointAlign.hpp"
#include "lvr2/registration/Metascan.hpp"

#include <iomanip>

using namespace std;

namespace lvr2
{

SLAMAlign::SLAMAlign(const SLAMOptions& options, const vector<SLAMScanPtr>& scans)
    : m_options(options), m_scans(scans), m_graph(&m_options), m_foundLoop(false), m_loopIndexCount(0)
{
    // The first Scan is never changed
    m_alreadyMatched = 1;

    for (auto& scan : m_scans)
    {
        reduceScan(scan);
    }
}

SLAMAlign::SLAMAlign(const SLAMOptions& options)
    : m_options(options), m_graph(&m_options), m_foundLoop(false), m_loopIndexCount(0)
{
    // The first Scan is never changed
    m_alreadyMatched = 1;
}

void SLAMAlign::setOptions(const SLAMOptions& options)
{
    m_options = options;
}

SLAMOptions& SLAMAlign::options()
{
    return m_options;
}

const SLAMOptions& SLAMAlign::options() const
{
    return m_options;
}

void SLAMAlign::addScan(const SLAMScanPtr& scan, bool match)
{
    reduceScan(scan);
    m_scans.push_back(scan);

    if (match)
    {
        this->match();
    }
}

void SLAMAlign::addScan(const ScanPtr& scan, bool match)
{
    addScan(make_shared<SLAMScanWrapper>(scan));
}

SLAMScanPtr SLAMAlign::scan(size_t index) const
{
    return m_scans[index];
}

void SLAMAlign::reduceScan(const SLAMScanPtr& scan)
{
    size_t prev = scan->numPoints();
    if (m_options.reduction >= 0)
    {
        scan->reduce(m_options.reduction, m_options.maxLeafSize);
    }
    if (m_options.minDistance >= 0)
    {
        scan->setMinDistance(m_options.minDistance);
    }
    if (m_options.maxDistance >= 0)
    {
        scan->setMaxDistance(m_options.maxDistance);
    }

    if (scan->numPoints() < prev)
    {
        scan->trim();

        if (m_options.verbose)
        {
            cout << "Removed " << (prev - scan->numPoints()) << " / " << prev << " Points -> " << scan->numPoints() << " left" << endl;
        }
    }
}

void SLAMAlign::match()
{
    // need at least 2 Scans
    if (m_scans.size() <= 1 || m_options.icpIterations <= 0)
    {
        return;
    }

    if (m_options.metascan && !m_metascan)
    {
        Metascan* meta = new Metascan();
        for (size_t i = 0; i < m_alreadyMatched; i++)
        {
            meta->addScan(m_scans[i]);
        }

        m_metascan = SLAMScanPtr(meta);
    }

    string scan_number_string = to_string(m_scans.size() - 1);

    // only match everything after m_alreadyMatched
    for (; m_alreadyMatched < m_scans.size(); m_alreadyMatched++)
    {
        cout << m_scans.size() << endl;
        size_t i = m_alreadyMatched;

        if (m_options.verbose)
        {
            cout << "Iteration " << setw(scan_number_string.length()) << i << "/" << scan_number_string << ": " << endl;
        }
        else
        {
            cout << setw(scan_number_string.length()) << i << "/" << scan_number_string << ": " << flush;
        }

        SLAMScanPtr prev = m_options.metascan ? m_metascan : m_scans[i - 1];
        const SLAMScanPtr& cur = m_scans[i];

        if (!m_options.trustPose && i != 1) // no deltaPose on first run
        {
            applyTransform(cur, prev->deltaPose());
        }
        else
        {
            if (m_options.createFrames)
            {
                applyTransform(cur, Matrix4d::Identity());
            }
        }

        ICPPointAlign icp(prev, cur);
        icp.setMaxMatchDistance(m_options.icpMaxDistance);
        icp.setMaxIterations(m_options.icpIterations);
        icp.setMaxLeafSize(m_options.maxLeafSize);
        icp.setEpsilon(m_options.epsilon);
        icp.setVerbose(m_options.verbose);

        icp.match();

        if (m_options.createFrames)
        {
            applyTransform(cur, Matrix4d::Identity());
        }

        if (m_options.metascan)
        {
            ((Metascan*)m_metascan.get())->addScan(cur);
        }

        checkLoopClose(i);
    }
}

void SLAMAlign::applyTransform(SLAMScanPtr scan, const Matrix4d& transform)
{
    scan->transform(transform, m_options.createFrames);

    if (m_options.createFrames)
    {
        bool found = false;
        for (const SLAMScanPtr& s : m_scans)
        {
            if (s != scan)
            {
                s->addFrame(found ? FrameUse::INVALID : FrameUse::UNUSED);
            }
            else
            {
                found = true;
            }
        }
    }
}

void SLAMAlign::checkLoopClose(size_t last)
{
    if (!m_options.doLoopClosing && !m_options.doGraphSLAM)
    {
        return;
    }

    bool hasLoop = false;
    size_t first = 0;

    vector<size_t> others;
    if (findCloseScans(m_scans, last, m_options, others))
    {
        hasLoop = true;
        first = others[0];
    }

    if (hasLoop && (m_loopIndexCount % 10 == 3) && m_options.doLoopClosing)
    {
        loopClose(first, last);
    }
    if (!hasLoop && m_loopIndexCount > 0 && m_options.doLoopClosing && !m_options.doGraphSLAM)
    {
        loopClose(first, last);
    }

    // falling edge
    if (m_foundLoop && !hasLoop && m_options.doGraphSLAM)
    {
        graphSLAM(last);
    }

    if (hasLoop)
    {
        m_loopIndexCount++;
    }
    else
    {
        m_loopIndexCount = 0;
    }
    m_foundLoop = hasLoop;
}

void SLAMAlign::loopClose(size_t first, size_t last)
{
    cout << "Loopclose " << first << " -> " << last << endl;

    Metascan* metaFirst = new Metascan();
    Metascan* metaLast = new Metascan();
    for (size_t i = 0; i < 3; i++)
    {
        metaFirst->addScan(m_scans[first + i]);
        metaLast->addScan(m_scans[last - i]);
    }

    SLAMScanPtr scanFirst(metaFirst);
    SLAMScanPtr scanLast(metaLast);

    ICPPointAlign icp(scanFirst, scanLast);
    icp.setMaxMatchDistance(m_options.slamMaxDistance);
    icp.setMaxIterations(m_options.slamIterations);
    icp.setMaxLeafSize(m_options.maxLeafSize);
    icp.setEpsilon(m_options.slamEpsilon);
    icp.setVerbose(m_options.verbose);

    Matrix4d transform = icp.match();

    for (size_t i = first + 3; i <= last - 3; i++)
    {
        double factor = (i - first) / (double)(last - first);

        Matrix4d delta = (transform - Matrix4d::Identity()) * factor + Matrix4d::Identity();

        m_scans[i]->transform(delta, m_options.createFrames, FrameUse::LOOPCLOSE);
    }

    if (m_options.createFrames)
    {
        // Add frame to unaffected scans
        for (size_t i = 0; i < 3; i++)
        {
            m_scans[first + i]->addFrame(FrameUse::LOOPCLOSE);
            m_scans[last - i]->addFrame(FrameUse::LOOPCLOSE);
        }
        for (size_t i = 0; i < first; i++)
        {
            m_scans[i]->addFrame();
        }
        for (size_t i = last - 2; i < m_scans.size(); i++)
        {
            m_scans[i]->addFrame(FrameUse::INVALID);
        }
    }
}

void SLAMAlign::graphSLAM(size_t last)
{
    m_graph.doGraphSLAM(m_scans, last);
}

void SLAMAlign::finish()
{
    match();

    if (m_options.doGraphSLAM)
    {
        graphSLAM(m_scans.size() - 1);
    }
}

} /* namespace lvr2 */
