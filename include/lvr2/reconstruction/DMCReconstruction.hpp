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

 /*
 * DMCReconstruction.hpp
 *
 *  Created on: 18.01.2019
 *      Author: Benedikt Schumacher
 */

#ifndef DMCReconstruction_HPP_
#define DMCReconstruction_HPP_

#include "lvr2/geometry/Vector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/reconstruction/PointsetMeshGenerator.hpp"
#include "lvr2/reconstruction/LocalApproximation.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/MCTable.hpp"
#include "lvr2/io/Progress.hpp"
#include "OctreeThreadPool.hpp"

#include "Octree.hpp"
#include "DualOctree.hpp"
#include "Location.hh"

using namespace __gnu_cxx;

namespace lvr2
{

struct my_dummy
{
    Location location;
    int next = -1;
};

static int MAX_LEVEL = 5;

/**
 * @brief A surface reconstruction object that implements the standard
 *        marching cubes algorithm using a octree and a thread pool for
 *        parallel computation.
 */
template<typename BaseVecT, typename BoxT>
class DMCReconstruction : public FastReconstructionBase<BaseVecT>, public PointsetMeshGenerator<BaseVecT>
{
public:

    /**
     * @brief Constructor.
     *
     * @param surface            Pointer to the surface
     * @param resolution         The number of intersections (on the longest side of the volume taken by the data points) used by the reconstruction.
     * @param isVoxelsize        If set to true, interpret resolution as voxelsize instead of number of intersections.
     * @param reconstructionType Type of the reconstruction (MC or PMC).
     * @param extrude            If set to true, missing children for each node will be created.
     */
    DMCReconstruction(
        PointsetSurfacePtr<BaseVecT> surface,
        BoundingBox<BaseVecT> bb,
        bool extrude);

    /**
     * @brief Destructor.
     */
    virtual ~DMCReconstruction();

    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<BaseVecT> &mesh);

    virtual void getMesh(
        BaseMesh<BaseVecT>& mesh,
        BoundingBox<BaseVecT>& bb,
        vector<unsigned int>& duplicates,
        float comparePrecision
    );

protected:

    /**
     * @brief Builds a tree level respectively creates the child nodes of a root node.
     *
     * @param parentPoints Points inside the parent node.
     * @param parent       Reference to the parent node.
     * @param childCenter  Centers of the children.
     * @param size         Actually voxelsize.
     * @param parentCenter Center of the parent node.
     */
    void buildTree(
        C_Octree<BaseVecT, BoxT, my_dummy> &parent,
        int levels);

    /**
     * @brief Traverses the octree and insert for each leaf the getSurface-function into the thread pool.
     *
     * @param mesh       The reconstructed mesh.
     * @param node       Actually node.
     * @param threadPool A thread pool.
     */
    void traverseTree(BaseMesh<BaseVecT> &mesh,
            C_Octree<BaseVecT, BoxT, my_dummy> &octree,
            OctreeThreadPool<BaseVecT, BoxT>* threadPool);

    void detectVertexForDualCell(
            C_Octree<BaseVecT, BoxT, my_dummy> &octree,
            CellHandle ch,
            int cells,
            float max_bb_width,
            uint pos,
            BaseVecT &feature);

    /**
     * @brief Performs a local reconstruction according to the standard Marching Cubes table from Paul Bourke.
     *
     * @param mesh The reconstructed mesh
     * @param leaf A octree leaf.
     */
    //void getSurface(BaseMesh<BaseVecT, BoxT> &mesh,
    //    OctreeLeaf<BaseVecT, BoxT> *leaf);

    void getSurface(BaseMesh<BaseVecT> &mesh,
        DualLeaf<BaseVecT, BoxT> *leaf,
        int cells);

    // The voxelsize used for reconstruction
    float m_voxelSize;

    // Counter of the edge points
    uint m_globalIndex;

    // Maximum voxelsize.
    float m_maxSize;

    // Sizes of Boundig Box
    BaseVecT bb_min;
    BaseVecT bb_max;
    float bb_size[3];
    float bb_longestSide;

    // Center of the bounding box.
    BaseVecT m_boundingBoxCenter;

    // Status of the extrusion.
    bool m_extrude;

    // Reconstructiontype
    string m_reconstructionType;

    // Count of the octree nodes.
    uint m_nodes;

    // Count of the octree nodes (extruded).
    uint m_nodesExtr;

    // Count of the octree leaves.
    uint m_leaves;

    // Count of the octree leaves (extruded):
    uint m_leavesExtr;

    // Count of the faces.
    uint m_faces;

    // Global mutex.
    boost::mutex m_globalMutex;

    // Mutex for adding faces to the mesh buffer.
    boost::mutex m_addMutex;

    // Global progress bar.
    ProgressBar *m_progressBar;

    // Pointer to the new Octree
    C_Octree<BaseVecT, BoxT, my_dummy> *octree;

    // Pointer to the thread pool.
    OctreeThreadPool<BaseVecT, BoxT> *m_threadPool;
};
} // namespace lvr2

#include "DMCReconstruction.tcc"

#endif /* DMCReconstruction_HPP_ */
