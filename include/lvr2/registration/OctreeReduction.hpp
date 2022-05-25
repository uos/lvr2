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
 * OctreeReduction.hpp
 *
 * @date Sep 23, 2019
 * @author Malte Hillmann
 * @author Thomas Wiemann
 * @author Justus Braun
 */

#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"

#include <vector>
#include <random>

namespace lvr2
{

// Informs the way a point is picked from a voxel
enum VoxelSamplingPolicy
{
    CLOSEST_TO_CENTER,
    RANDOM_SAMPLE
};

class OctreeReduction
{
public:
    /**
     * @brief Construct a new OctreeReduction object
     * 
     * @param pointBuffer The points to be reduced
     * @param voxelSize The minimum size of a Leaf Node.
     *                  Anything smaller will be condensed using the sampling policy
     * @param minPointsPerVoxel Leaf nodes with fewer points than this will be ignored
     * @param samplingPolicy The way a point is picked from a condensed Leaf node
     */
    OctreeReduction(PointBufferPtr& pointBuffer, float voxelSize, size_t minPointsPerVoxel, VoxelSamplingPolicy samplingPolicy = CLOSEST_TO_CENTER);
    /**
     * @brief Construct a new OctreeReduction object
     * 
     * @param points The points to be reduced. THIS DATA WILL BE MODIFIED IN PLACE.
     * @param n The length of points. Will be updated to the number of points after reduction.
     * @param voxelSize The minimum size of a Leaf Node.
     *                  Anything smaller will be condensed using the sampling policy
     * @param minPointsPerVoxel Leaf nodes with fewer points than this will be ignored
     * @param samplingPolicy The way a point is picked from a condensed Leaf node
     */
    OctreeReduction(Vector3f* points, size_t& n, float voxelSize, size_t minPointsPerVoxel, VoxelSamplingPolicy samplingPolicy = CLOSEST_TO_CENTER);

    /**
     * @brief Get the Reduced Points object. ONLY WORKS IF PointBufferPtr CONSTRUCTOR WAS USED
     * 
     * @return PointBufferPtr The reduced points
     */
    PointBufferPtr getReducedPoints();

private:
    /// initialize and run the reduction
    void init();

    /// recursive core function for reduction
    void createOctree(size_t start, size_t n, const Vector3f& min, const Vector3f& max, unsigned int level = 0);

    /// samples [start, start+n) according to m_samplingPolicy
    void sampleRange(size_t start, size_t n, const Vector3f& center);

    size_t splitPoints(size_t start, size_t n, unsigned int axis, float splitValue);

    float               m_voxelSize;
    size_t              m_minPointsPerVoxel;
    size_t              m_numPoints; 
    bool*               m_flags; // m_flags[i] == true iff m_points[i] should be deleted
    Vector3f*           m_points; // not owned
    std::vector<size_t> m_pointIndices;
    PointBufferPtr      m_pointBuffer;
    VoxelSamplingPolicy m_samplingPolicy;
    std::mt19937        m_randomEngine;
};

/**
 * @brief Reference implementation of an octree-based reduction algorithm
 * 
 */
class OctreeReductionAlgorithm : public ReductionAlgorithm
{
public:
    OctreeReductionAlgorithm(double voxelSize, size_t minPoints) : 
        m_octree(nullptr), m_voxelSize(voxelSize), m_minPoints(minPoints) {};

    void setPointBuffer(PointBufferPtr ptr) override
    {
        // Create octree
        m_octree.reset(new OctreeReduction(ptr, m_voxelSize, m_minPoints, VoxelSamplingPolicy::RANDOM_SAMPLE));
    }

    PointBufferPtr getReducedPoints()
    {
        if(m_octree)
        {
            return m_octree->getReducedPoints();
        }
        return PointBufferPtr(new PointBuffer());
    }

private:
    std::shared_ptr<OctreeReduction> m_octree;
    double m_voxelSize;
    size_t m_minPoints;
};

} // namespace lvr2

#endif