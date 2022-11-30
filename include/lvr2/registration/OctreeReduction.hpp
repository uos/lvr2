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
#include "lvr2/util/Logging.hpp"

#include <vector>

namespace lvr2
{

class OctreeReductionBase
{
public:
    /// @brief Default constructor is not available
    OctreeReductionBase() = delete;

    /**
     * @brief Construct a new Octree Reduction Base object
     * 
     * @param pointBuffer       Point buffer with the initial point cloud 
     * @param voxelSize         Voxel size of the octree
     * @param maxPointsPerVoxel Maximum points per voxel
     */
    OctreeReductionBase(PointBufferPtr pointBuffer, float voxelSize, size_t maxPointsPerVoxel) 
        : m_voxelSize(voxelSize), 
          m_maxPointsPerVoxel(maxPointsPerVoxel), 
          m_numPoints(pointBuffer->numPoints()), 
          m_pointBuffer(pointBuffer) {}


    /**
     * @brief Get the reduced point buffer 
     */
    virtual PointBufferPtr getReducedPoints() = 0;

protected:
    /// Indices of the points that are kept after reduction
    std::vector<size_t> m_pointIndices;

    /// Point buffer
    PointBufferPtr      m_pointBuffer;

    /// Voxel size
    float               m_voxelSize;

    /// Max number of points per voxel
    size_t              m_maxPointsPerVoxel;

    /// Number of points in point buffer
    size_t              m_numPoints; 
};

class RandomSampleOctreeReduction : public OctreeReductionBase
{
public:
    /**
     * @brief Constructs an random-sampling based octree that holds up to \ref maxPointsPerVoxel
     *        Randomly selected points per voxel 
     * 
     * @param pointBuffer       The point buffer with the initial point cloud
     * @param voxelSize         Minimum size of a Leaf Node. Anything smaller will be 
     *                          condensed using the random ampling policy
     * @param maxPointsPerVoxel Maximum number of points per voxel
     */
    RandomSampleOctreeReduction(PointBufferPtr pointBuffer, float voxelSize, size_t maxPointsPerVoxel);
    /**
     * @brief Constructs an random-sampling octree from the given point array
     * 
     * @param points            The points to be reduced. THIS DATA WILL BE MODIFIED IN PLACE.
     * @param n                 The length of points. Will be updated to the number of points after 
     *                          reduction.
     * @param voxelSize         The minimum size of a Leaf Node. Anything smaller will be 
     *                          condensed using the sampling policy
     * @param maxPointsPerVoxel Maximum number of points per voxel
     */
    RandomSampleOctreeReduction(Vector3f* points, size_t& n, float voxelSize, size_t maxPointsPerVoxel);

    /**
     * @brief Get the Reduced Points object. ONLY WORKS IF PointBufferPtr CONSTRUCTOR WAS USED
     * 
     * @return PointBufferPtr The reduced points
     */
    PointBufferPtr getReducedPoints() override;

private:
    /// Initialize and run the reduction
    void init();

    /// @brief Recursive helper function to build the octree
    /// @param start            Start index of the points to distribute
    /// @param n                Number of points to distribute
    /// @param min              Min coordinates of current bounding volume
    /// @param max              Max coordinates of current bounding volume
    /// @param level            Octree depth
    void createOctree(size_t start, size_t n, const Vector3f& min, const Vector3f& max, unsigned int level = 0);

    /// @brief Sorts the leaf points according to split value 
    /// @param start            Start index of the points
    /// @param n                Number of points to distribute
    /// @param axis             Split axis
    /// @param splitValue       Split valie
    /// @return 
    size_t splitPoints(size_t start, size_t n, unsigned int axis, float splitValue);

    /// m_flags[i] == true iff m_points[i] should be deleted
    bool*               m_flags; 

    /// Pointer to handled point array (not owned)
    Vector3f*           m_points; 
};

class NearestCenterOctreeReduction : public OctreeReductionBase
{
      /**
     * @brief Constructs octtree that holds up to \ref maxPointsPerVoxel
     *        points per voxel. It selects one point per voxel which is 
     *        closest to the voxel center
     * 
     * @param pointBuffer       The point buffer with the initial point cloud
     * @param voxelSize         Minimum size of a Leaf Node. Anything smaller will be 
     *                          condensed using the random ampling policy
     * @param maxPointsPerVoxel Maximum number of points per voxel
     */
    NearestCenterOctreeReduction(PointBufferPtr pointBuffer, float voxelSize, size_t maxPointsPerVoxel);

    /**
     * @brief Get the Reduced Points object. ONLY WORKS IF PointBufferPtr CONSTRUCTOR WAS USED
     * 
     * @return PointBufferPtr The reduced points
     */
    PointBufferPtr getReducedPoints() override;    
};

/**
 * @brief Reference implementation of an octree-based reduction algorithm
 * 
 */
class OctreeReductionAlgorithm : public ReductionAlgorithm
{
public:
    OctreeReductionAlgorithm(float voxelSize, size_t maxPoints) : 
        m_octree(nullptr), m_voxelSize(voxelSize), m_maxPoints(maxPoints) {};

    void setPointBuffer(PointBufferPtr ptr) override
    {
        // Create octree
        m_octree.reset(new RandomSampleOctreeReduction(ptr, m_voxelSize, m_maxPoints));
    }

    PointBufferPtr getReducedPoints() override
    {
        // Check if an octree instance exists
        // Otherwise return empty point buffer
        if(m_octree)
        {
            return m_octree->getReducedPoints();
        }
        lvr2::logout::get() 
            << lvr2::warning << "[OctreeReduction] Cannot get reduced points without point buffer." << lvr2::endl;

        return PointBufferPtr(new PointBuffer());
    }

private:
    /// Pointer to the octree that is used for reduction
    std::shared_ptr<OctreeReductionBase> m_octree;

    /// Voxel size  
    float m_voxelSize;

    /// Maximum number of points
    size_t m_maxPoints;
};

} // namespace lvr2

#endif