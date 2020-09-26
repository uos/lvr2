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

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/reconstruction/PointsetMeshGenerator.hpp"
#include "lvr2/reconstruction/LocalApproximation.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/MCTable.hpp"
#include "lvr2/io/Progress.hpp"
#include "DMCVecPointHandle.hpp"

#include "Octree.hpp"
#include "DualOctree.hpp"
#include "Location.hh"
#include "lvr2/reconstruction/FastReconstruction.hpp"

namespace lvr2
{

struct my_dummy
{
    Location location;
    int next = -1;
};

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
     * @param surface       Pointer to the surface
     * @param bb            BoundingBox of the PointCloud
     * @param dual          
     * @param maxLevel      Max allowed octree level
     * @param maxError      Max allowed error between points and surfaces
     */
    DMCReconstruction(
        PointsetSurfacePtr<BaseVecT> surface,
        BoundingBox<BaseVecT> bb,
        bool dual,
        int maxLevel,
        float maxError);

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
     * @brief Builds a dual cell at given position
     *
     * @param cell   Cell of the regual octree
     * @param cells  number of cells at the cells octree level
     * @param octree Reference to the octree
     * @param pos    Position of the specific dual cell
     */
    DualLeaf<BaseVecT, BoxT>* getDualLeaf(
        CellHandle &cell,
        int cells,
        C_Octree<BaseVecT, BoxT, my_dummy> &octree,
        char pos);

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
        int levels,
        bool dual);

    /**
     * @brief Traverses the octree and insert for each leaf the getSurface-function into the thread pool.
     *
     * @param mesh       The reconstructed mesh.
     * @param node       Actually node.
     */
    void traverseTree(BaseMesh<BaseVecT> &mesh,
        C_Octree<BaseVecT, BoxT, my_dummy> &octree);

    /**
     * @brief Calculates the position of a aspecific point in a dual cell
     *
     * @param octree The current octree instace
     * @param ch The current cellHandle of the octree
     * @param cells Number of possible cells at current level
     * @param max_bb_width Width of the bounding box
     * @param pos Position of the dual cell relative to the given cell
     * @param onEdge Indicator whether the wished position lies on an edge
     * @param feature Vector that should be filles with the correct position
     */
    void detectVertexForDualCell(
        C_Octree<BaseVecT, BoxT, my_dummy> &octree,
        CellHandle ch,
        int cells,
        float max_bb_width,
        uint pos,
        int onEdge,
        BaseVecT &feature);

    /**
     * @brief Calculates a rotation matrix for a triangle that rotates it into xy
     *
     * @param matrix Array the matrix should be written in
     * @param v1 First point of the triangle
     * @param v2 Second point of the triangle
     * @param v3 Third point of the triangle
     */
    void getRotationMatrix(
        float matrix[9],
        BaseVecT v1,
        BaseVecT v2,
        BaseVecT v3);

    /**
     * @brief Calculates the distance between a point and a triangle
     *
     * @param p Vertex to calculate distance for
     * @param v1 First point of the triangle
     * @param v2 Second point of the triangle
     * @param v3 Third point of the triangle
     */
    float getDistance(BaseVecT p,
        BaseVecT v1,
        BaseVecT v2,
        BaseVecT v3);

    /**
     * @brief Calculates the distance between a point and a line
     *
     * @param p Vertex to calculate distance for
     * @param v1 First point of the line
     * @param v2 Second point of the line
     */
    float getDistance(BaseVecT p,
        BaseVecT v1,
        BaseVecT v2);

    /**
     * @brief Calculates the distance between to points
     *
     * @param p Vertex to calculate distance for
     * @param v1 Point
     */
    float getDistance(BaseVecT v1,
        BaseVecT v2);

    /**
     * @brief Calculates whether the given vertex lies left, right or on the given line
     *
     * @param p Vertex to check position for
     * @param v1 First point of the line
     * @param v2 Second point of the line
     */
    float edgeEquation(BaseVecT p,
        BaseVecT v1,
        BaseVecT v2);

    /**
     * @brief Performs a matrix multiplication
     *
     * @param matrix Pointer to the matrix
     * @param vector Pointer to th vector
     */
    void matrixDotVector(float* matrix,
        BaseVecT* vector);

    /**
     * @brief Performs a local reconstruction according to the standard Marching Cubes table from Paul Bourke.
     *
     * @param mesh The reconstructed mesh
     * @param leaf A octree leaf.
     * @param cells
     */
    void getSurface(BaseMesh<BaseVecT> &mesh,
        DualLeaf<BaseVecT, BoxT> *leaf,
        int cells,
        short level);

    /**
     * @brief Saves the octree as wireframe. WORKS ONLY SINGLE THREADED!
     *
     * @param parent The octree
     */
    void drawOctree(C_Octree<BaseVecT, BoxT, my_dummy> &parent);

    // Indicator whether the point fitting shoulb be done on dual cells
    bool m_dual;

    // The maximum allowed level for the octree
    int m_maxLevel;

    // The max allowed arror between points and surfaces
    float m_maxError;

    // Sizes of Boundig Box
    BaseVecT bb_min;
    BaseVecT bb_max;
    float bb_size[3];
    float bb_longestSide;

    // Center of the bounding box.
    BaseVecT m_boundingBoxCenter;

    // Count of the octree leaves.
    uint m_leaves;

    // Global progress bar.
    ProgressBar *m_progressBar;

    // Pointer to the new Octree
    C_Octree<BaseVecT, BoxT, my_dummy> *octree;

    // PointHandler
    unique_ptr<DMCPointHandle<BaseVecT>> m_pointHandler;

    // just for visualization
    std::vector< BaseVecT > dualVertices;
};
} // namespace lvr2

#include "DMCReconstruction.tcc"

#endif /* DMCReconstruction_HPP_ */
