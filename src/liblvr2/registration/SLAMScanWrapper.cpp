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
 * SLAMScanWrapper.cpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */

#include "lvr2/registration/SLAMScanWrapper.hpp"
#include "lvr2/registration/OctreeReduction.hpp"

#include <fstream>

namespace lvr2
{

SLAMScanWrapper::SLAMScanWrapper(ScanPtr scan)
    : m_scan(scan), m_deltaPose(Transformd::Identity())
{
    if (m_scan)
    {
        m_scan->transformation = m_scan->poseEstimation;

        // TODO: m_scan->m_points->load();

        m_numPoints = m_scan->points->numPoints();
        lvr2::floatArr arr = m_scan->points->getPointArray();

        m_points.resize(m_numPoints);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m_numPoints; i++)
        {
            m_points[i] = Vector3f(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]);
        }

        // TODO: m_scan->m_points->unload();
        m_scan->points.reset();
    }
    else
    {
        m_numPoints = 0;
    }
}

ScanPtr SLAMScanWrapper::innerScan()
{
    return m_scan;
}

void SLAMScanWrapper::transform(const Transformd& transform, bool writeFrame, FrameUse use)
{
    m_scan->transformation = transform * m_scan->transformation;
    m_deltaPose = transform * m_deltaPose;

    if (writeFrame)
    {
        addFrame(use);
    }
}

void SLAMScanWrapper::reduce(double voxelSize, int maxLeafSize)
{
    RandomSampleOctreeReduction reduction(m_points.data(), m_numPoints, voxelSize, maxLeafSize);
    m_points.resize(m_numPoints);
}

void SLAMScanWrapper::setMinDistance(double minDistance)
{
    double sqDist = minDistance * minDistance;

    size_t cur = 0;
    while (cur < m_numPoints)
    {
        if (m_points[cur].squaredNorm() <= sqDist)
        {
            m_points[cur] = m_points[m_numPoints - 1];
            m_numPoints--;
        }
        else
        {
            cur++;
        }
    }
    m_points.resize(m_numPoints);
}

void SLAMScanWrapper::setMaxDistance(double maxDistance)
{
    double sqDist = maxDistance * maxDistance;

    size_t cur = 0;
    while (cur < m_numPoints)
    {
        if (m_points[cur].squaredNorm() >= sqDist)
        {
            m_points[cur] = m_points[m_numPoints - 1];
            m_numPoints--;
        }
        else
        {
            cur++;
        }
    }
    m_points.resize(m_numPoints);
}

void SLAMScanWrapper::trim()
{
    m_points.resize(m_numPoints);
    m_points.shrink_to_fit();
}

Vector3d SLAMScanWrapper::point(size_t index) const
{
    const Vector3f& p = m_points[index];
    Vector4d extended(p.x(), p.y(), p.z(), 1.0);
    return (pose() * extended).block<3, 1>(0, 0);
}

const Vector3f& SLAMScanWrapper::rawPoint(size_t index) const
{
    return m_points[index];
}

size_t SLAMScanWrapper::numPoints() const
{
    return m_numPoints;
}

const Transformd& SLAMScanWrapper::pose() const
{
    return m_scan->transformation;
}

const Transformd& SLAMScanWrapper::deltaPose() const
{
    return m_deltaPose;
}

const Transformd& SLAMScanWrapper::initialPose() const
{
    return m_scan->poseEstimation;
}

Vector3d SLAMScanWrapper::getPosition() const
{
    return pose().block<3, 1>(0, 3);
}

void SLAMScanWrapper::addFrame(FrameUse use)
{
    m_frames.push_back(make_pair(pose(), use));
}

size_t SLAMScanWrapper::frameCount() const
{
    return m_frames.size();
}

const std::pair<Transformd, FrameUse>& SLAMScanWrapper::frame(size_t index) const
{
    return m_frames[index];
}

void SLAMScanWrapper::writeFrames(std::string path) const
{
    std::ofstream out(path);
    for (const std::pair<Transformd, FrameUse>& frame : m_frames)
    {
        for (int i = 0; i < 16; i++)
        {
            out << frame.first(i) << " ";
        }

        out << (int)frame.second << std::endl;
    }
}

KDTreePtr<Vector3f> SLAMScanWrapper::createKDTree(size_t maxLeafSize) const
{
    std::unique_ptr<Vector3f[]> points(new Vector3f[m_numPoints]);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m_numPoints; i++)
    {
        points[i] = point(i).cast<float>();
    }

    return std::move(KDTree<Vector3f>::create(std::move(points), m_numPoints, maxLeafSize));
}

size_t SLAMScanWrapper::nearestNeighbors(
    KDTreePtr<Vector3f> tree, SLAMScanPtr scan,
    std::vector<Vector3f*>& neighbors, double maxDistance,
    Vector3d& centroid_m, Vector3d& centroid_d)
{
    size_t found = nearestNeighbors(tree, scan, neighbors, maxDistance);

    centroid_m = Vector3d::Zero();
    centroid_d = Vector3d::Zero();

    for (size_t i = 0; i < scan->numPoints(); i++)
    {
        if (neighbors[i] != nullptr)
        {
            centroid_m += neighbors[i]->cast<double>();
            centroid_d += scan->point(i);
        }
    }

    centroid_m /= found;
    centroid_d /= found;

    return found;
}

size_t SLAMScanWrapper::nearestNeighbors(KDTreePtr<Vector3f> tree, SLAMScanPtr scan, std::vector<Vector3f*>& neighbors, double maxDistance)
{
    if (neighbors.size() < scan->numPoints())
    {
        neighbors.resize(scan->numPoints());
    }
    size_t found = 0;
    double distance = 0.0;

    #pragma omp parallel for firstprivate(distance) reduction(+:found) schedule(dynamic,8)
    for (size_t i = 0; i < scan->numPoints(); i++)
    {
        if (tree->nnSearch(scan->point(i), neighbors[i], distance, maxDistance))
        {
            found++;
        }
    }

    return found;
}

} /* namespace lvr2 */
