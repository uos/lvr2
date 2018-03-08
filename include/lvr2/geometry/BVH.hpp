/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
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

#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/Vector.hpp>

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

        Point<BaseVecT> center;
        Normal<BaseVecT> normal;

        // intersection pre-computed cache
        float d, d1, d2, d3;
        Normal<BaseVecT> e1, e2, e3;

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

#include <lvr2/geometry/BVH.tcc>
