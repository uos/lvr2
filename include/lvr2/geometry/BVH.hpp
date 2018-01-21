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
struct BVHNode {
    BoundingBox<BaseVecT> bb;
    virtual bool isLeaf() = 0;
};

template<typename BaseVecT>
struct BVHInner: BVHNode<BaseVecT> {
    unique_ptr<BVHNode<BaseVecT>> left;
    unique_ptr<BVHNode<BaseVecT>> right;
    virtual bool isLeaf() { return false; }
};

template<typename BaseVecT>
struct BVHLeaf: BVHNode<BaseVecT> {
    vector<float> triangles;
    virtual bool isLeaf() { return true; }
};

template<typename BaseVecT>
class BVHTree
{
public:
    BVHTree(const vector<float>& vertices, const vector<uint32_t>& faces);
private:
    BVHInner<BaseVecT> m_root;
    BVHInner<BaseVecT> buildTree(const vector<float>& vertices, const vector<uint32_t>& faces);
};

} /* namespace lvr2 */

#include <lvr2/geometry/BVH.tcc>
