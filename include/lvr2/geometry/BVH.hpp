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

using std::unique_ptr;
using std::vector;

namespace lvr2
{

template<typename BaseVecT>
struct Triangle {

    Triangle();

    // indices in vertex array
    uint32_t idx1;
    uint32_t idx2;
    uint32_t idx3;

    Point<BaseVecT> center;
    Normal<BaseVecT> normal;

    // intersection pre-computed cache
    float d, d1, d2, d3;
    Normal<BaseVecT> e1, e2, e3;

    BoundingBox<BaseVecT> bb;
};

template<typename BaseVecT>
struct AABB {
    BoundingBox<BaseVecT> bb;
    vector<Triangle<BaseVecT>> triangles;
};

template<typename BaseVecT>
struct BVHNode {
    BoundingBox<BaseVecT> bb;
    virtual bool isLeaf() = 0;
};

template<typename BaseVecT>
using BVHNodePtr = unique_ptr<BVHNode<BaseVecT>>;

template<typename BaseVecT>
struct BVHInner: BVHNode<BaseVecT> {
    unique_ptr<BVHNode<BaseVecT>> left;
    unique_ptr<BVHNode<BaseVecT>> right;
    virtual bool isLeaf() { return false; }
};

template<typename BaseVecT>
using BVHInnerPtr = unique_ptr<BVHInner<BaseVecT>>;

template<typename BaseVecT>
struct BVHLeaf: BVHNode<BaseVecT> {
    vector<Triangle<BaseVecT>> triangles;
    virtual bool isLeaf() { return true; }
};

template<typename BaseVecT>
using BVHLeafPtr = unique_ptr<BVHLeaf<BaseVecT>>;

template<typename BaseVecT>
class BVHTree
{
public:
    BVHTree(const vector<float>& vertices, const vector<uint32_t>& faces);
private:
    using AABB_t = AABB<BaseVecT>;

    BVHNodePtr<BaseVecT> m_root;
    BVHNodePtr<BaseVecT> buildTree(const vector<float>& vertices, const vector<uint32_t>& faces);
    BVHNodePtr<BaseVecT> buildTreeRecursive(vector<AABB_t>& work, uint32_t depth = 0);
};

} /* namespace lvr2 */

#include <lvr2/geometry/BVH.tcc>
