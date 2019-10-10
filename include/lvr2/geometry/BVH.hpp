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
 * BVH.hpp
 *
 *  @date 21.01.2018
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#pragma once

#include <vector>
#include <memory>

#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/geometry/Normal.hpp"

using std::unique_ptr;
using std::vector;
using std::pair;

namespace lvr2
{

/**
 * @brief Implementation of an Bounding Volume Hierarchy Tree used for ray casting
 *
 * This class generates a BVHTree from the given triangle mesh represented by vertices and faces. AABB are used as
 * bounding volumes. The Tree Contains inner nodes and leaf nodes. The leaf nodes are grouped into inner nodes using
 * the surface area heuristic. The tree is represented in two ways: the normal tree structure and the cache friendly
 * index representation.
 *
 * @tparam BaseVecT
 */
template<typename BaseVecT>
class BVHTree
{
public:

    /**
     * @brief Constructs the tree itself and it's cache friendly representation
     *
     * @param vertices Vertices of mesh to create tree for
     * @param faces Faces of mesh to create tree for
     */
    BVHTree(const vector<float>& vertices, const vector<uint32_t>& faces);

    /**
     * @brief 
     */
    BVHTree(
        const floatArr vertices, size_t n_vertices,
        const indexArray faces, size_t n_faces
    );

    /**
     * @brief
     */
    BVHTree(const MeshBufferPtr mesh);

    /**
     * @return Index list (for getTrianglesIntersectionData) of triangles in the leaf nodes
     */
    const vector<uint32_t>& getTriIndexList() const;

    /**
     * @return Boundaries of the the AABB of all nodes
     */
    const vector<float>& getLimits() const;

    /**
     * @return Four values per node:
     *      1. value:   1st bit of first value indicates, wether the current node is a leaf node or inner node.
     *                  If it's a leaf node the rest of the bits indicate the number of triangles in this node.
     *      2. value:   Only set, if the node is an inner node. Indicates the start index of the left sub tree in this
     *                  vector.
     *      3. value:   Only set, if the node is an inner node. Indicates the start index of the right sub tree in this
     *                  vector.
     *      4. value:   Only set, if the node is a leaf node. Contains the start index for triangles in getTriIndexList
     */
    const vector<uint32_t>& getIndexesOrTrilists() const;

    /**
     * @brief Returns precalculated values for the triangle intersection tests
     *
     * @return 16 values per triangle:
     *      1-3: x, y, z of normal
     *      4: normal.dot(point1)
     *      5-7: x, y, z of edge plane vector 1
     *      8: scalar product of point 1 and edge plane vector 1
     *      9-11: x, y, z of edge plane vector 2
     *      12: scalar product of point 2 and edge plane vector 2
     *      13-15: x, y, z of edge plane vector 3
     *      16: scalar product of point 3 and edge plane vector 3
     *
     */
    const vector<float>& getTrianglesIntersectionData() const;

private:

    // Internal triangle representation
    struct Triangle {

        Triangle();

        // indices in vertex array
        // todo: not used?
        uint32_t idx1;
        uint32_t idx2;
        uint32_t idx3;

        BaseVecT center;
        Normal<float> normal;

        // intersection pre-computed cache
        float d, d1, d2, d3;
        Normal<float> e1, e2, e3;

        // todo: not used?
        BoundingBox<BaseVecT> bb;
    };

    // Internal AABB representation
    struct AABB {
        BoundingBox<BaseVecT> bb;

        // Triangles corresponding to this AABB
        vector<size_t> triangles;
    };

    // Abstract tree node
    struct BVHNode {
        BoundingBox<BaseVecT> bb;
        virtual bool isLeaf() = 0;
    };
    using BVHNodePtr = unique_ptr<BVHNode>;

    // Inner tree node
    struct BVHInner: BVHNode {

        // Left sub tree
        BVHNodePtr left;

        // Right sub tree
        BVHNodePtr right;
        virtual bool isLeaf() { return false; }
    };
    using BVHInnerPtr = unique_ptr<BVHInner>;

    // Leaf tree node
    struct BVHLeaf: BVHNode {

        // Triangles corresponding to this leaf node
        vector<size_t> triangles;
        virtual bool isLeaf() { return true; }
    };
    using BVHLeafPtr = unique_ptr<BVHLeaf>;

    // working variables for tree construction
    BVHNodePtr m_root;
    vector<Triangle> m_triangles;

    // cache friendly data for the SIMD device
    vector<uint32_t> m_triIndexList;
    vector<float> m_limits;
    vector<uint32_t> m_indexesOrTrilists;
    vector<float> m_trianglesIntersectionData;

    /**
     * @brief Builds the tree without it's cache friendly representation. Utilizes the buildTreeRecursive method.
     *
     * @param vertices Vertices of mesh to create tree for
     * @param faces Faces of mesh to create tree for
     *
     * @return Root node of the tree
     */
    BVHNodePtr buildTree(const vector<float>& vertices, const vector<uint32_t>& faces);

    /**
     * @brief Builds the tree without it's cache friendly representation. Utilizes the buildTreeRecursive method.
     *
     * @param vertices Vertices of mesh to create tree for
     * @param faces Faces of mesh to create tree for
     *
     * @return Root node of the tree
     */
    BVHNodePtr buildTree(
        const floatArr vertices, size_t n_vertices,
        const indexArray faces, size_t n_faces
    );

    /**
     * @brief Recursive method to build the tree.
     *
     * @param work The current work queue to generate the tree from
     * @param depth The current depth of the tree
     *
     * @return Root node of the current tree
     */
    BVHNodePtr buildTreeRecursive(vector<AABB>& work, uint32_t depth = 0);

    /**
     * @brief Creates the cache friendly representation of the tree. Needs the tree itself!
     */
    void createCFTree();

    /**
     * @brief Recursive method for cache friendly tree creation.
     *
     * @param currentNode Current node to convert to it's cache friendly form
     * @param idxBoxes Number of boxes created for the current tree. This has to be increased every time a box
     *        is processed.
     */
    void createCFTreeRecursive(BVHNodePtr currentNode, uint32_t& idxBoxes);

    /**
     * @brief Converts the precalculated triangle intersection data to a SIMD friendly structure
     */
    void convertTrianglesIntersectionData();
};

} /* namespace lvr2 */

#include "lvr2/geometry/BVH.tcc"
