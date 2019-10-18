#ifndef __OCTREE_REDUCTION__
#define __OCTREE_REDUCTION__

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
 * Metascan.hpp
 *
 *  @date Sep 23, 2019
 *  @author Thomas Wiemann
 *  @author Malte Hillmann
 */

#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <vector>

namespace lvr2
{

class OctreeReduction
{
public:
    OctreeReduction(PointBufferPtr& pointBuffer, const double& voxelSize, const size_t& minPointsPerVoxel);
    OctreeReduction(Vector3f* points, const size_t& n, const double& voxelSize, const size_t& minPointsPerVoxel);

    PointBufferPtr getReducedPoints();
    void getReducedPoints(Vector3f& points, size_t& n);

    ~OctreeReduction() { delete[] m_flags;}

private:
    template<typename T>
    void createOctree(T* points, const int& n, bool* flagged, const T& min, const T& max, const int& level);

    template<typename T>
    size_t splitPoints(T* points, const size_t& n, const int axis, const double& splitValue);

    template<typename T>
    void createOctree(lvr2::PointBufferPtr& points, size_t s, size_t n, bool* flagged, const lvr2::Vector3<T>& min, const lvr2::Vector3<T>& max, const int& level);

    template<typename T>
    size_t splitPoints(lvr2::PointBufferPtr& points, size_t s, size_t n, const int axis, const double& splitValue);

    template<typename T>
    void swapAllChannelsOfType(lvr2::PointBufferPtr& points, const size_t& l, const size_t& r);

    template<typename T>
    void swapInChannel(lvr2::Channel<T>& ch, const size_t& l, const size_t& r);

    double  m_voxelSize;
    size_t  m_minPointsPerVoxel;
    size_t  m_numPoints; 
    bool*   m_flags;

    PointBufferPtr  m_pointBuffer;
    Vector3f        m_points;
};

} // namespace lvr2

#include "lvr2/registration/OctreeReduction.tcc"

#endif