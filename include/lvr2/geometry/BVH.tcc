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
 * BVH.tcc
 *
 *  @date 21.01.2018
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <limits>

using std::make_unique;

namespace lvr2
{

template<typename BaseVecT>
Triangle<BaseVecT>::Triangle()
    : idx1(0)
    , idx2(0)
    , idx3(0)
    , center(0, 0, 0)
    , normal(0, 1, 0)
    , d(0.0f)
    , d1(0.0f)
    , d2(0.0f)
    , d3(0.0f)
    , e1(0, 1, 0)
    , e2(0, 1, 0)
    , e3(0, 1, 0)
    , bb()
{

}

template<typename BaseVecT>
BVHTree<BaseVecT>::BVHTree(const vector<float>& vertices, const vector<uint32_t>& faces)
{
    m_root = buildTree(vertices, faces);
}

template<typename BaseVecT>
BVHNodePtr<BaseVecT> BVHTree<BaseVecT>::buildTree(const vector<float>& vertices, const vector<uint32_t>& faces)
{
    vector<AABB_t> work;
    work.reserve(faces.size() / 3);

    BoundingBox<BaseVecT> outerBb;

    for (size_t i = 0; i < faces.size(); i += 3)
    {
        Point<BaseVecT> point1;
        point1.x = vertices[faces[i]];
        point1.y = vertices[faces[i]+1];
        point1.z = vertices[faces[i]+2];

        Point<BaseVecT> point2;
        point2.x = vertices[faces[i+1]];
        point2.y = vertices[faces[i+1]+1];
        point2.z = vertices[faces[i+1]+2];

        Point<BaseVecT> point3;
        point3.x = vertices[faces[i+2]];
        point3.y = vertices[faces[i+2]+1];
        point3.z = vertices[faces[i+2]+2];

        BoundingBox<BaseVecT> faceBb;
        faceBb.expand(point1);
        faceBb.expand(point2);
        faceBb.expand(point3);

        Triangle<BaseVecT> triangle;
        triangle.bb = faceBb;
        triangle.center = Point<BaseVecT>((point1.asVector() + point2.asVector() + point3.asVector()) / 3.0f);
        triangle.idx1 = faces[i];
        triangle.idx2 = faces[i+1];
        triangle.idx3 = faces[i+2];

        BaseVecT vc1 = point2 - point1;
        BaseVecT vc2 = point3 - point2;
        BaseVecT vc3 = point1 - point3;

        // pick best normal
        BaseVecT normal1(vc1.cross(vc2));
        BaseVecT normal2(vc2.cross(vc3));
        BaseVecT normal3(vc3.cross(vc1));
        BaseVecT bestNormal = normal1;
        if (normal2.length() > bestNormal.length())
        {
            bestNormal = normal2;
        }
        if (normal3.length() > bestNormal.length())
        {
            bestNormal = normal3;
        }
        triangle.normal = Normal<BaseVecT>(bestNormal);

        triangle.d = triangle.normal.dot(point1);

        // edge plains
        triangle.e1 = Normal<BaseVecT>(triangle.normal.cross(vc1));
        triangle.d1 = triangle.e1.dot(point1);

        triangle.e2 = Normal<BaseVecT>(triangle.normal.cross(vc2));
        triangle.d2 = triangle.e2.dot(point2);

        triangle.e3 = Normal<BaseVecT>(triangle.normal.cross(vc3));
        triangle.d3 = triangle.e3.dot(point3);

        AABB_t aabb;
        aabb.bb = faceBb;
        aabb.triangles.push_back(triangle);

        outerBb.expand(faceBb);
        work.push_back(aabb);
    }

    std::cout << outerBb << " count: " << work.size() << std::endl;

    auto out = buildTreeRecursive(work);
    out->bb = outerBb;

    return out;
}

template<typename BaseVecT>
BVHNodePtr<BaseVecT> BVHTree<BaseVecT>::buildTreeRecursive(vector<AABB_t>& work, uint32_t depth)
{
    // terminate recursion, if work size is small enough
    if (work.size() < 4)
    {
        auto leaf = make_unique<BVHLeaf<BaseVecT>>();
        for (auto aabb: work)
        {
            leaf->triangles.push_back(aabb.triangles.front());
        }
        return leaf;
    }

    // divide node into smaller nodes
    BoundingBox<BaseVecT> bb;
    for (auto aabb: work)
    {
        bb.expand(aabb.bb);
    }

    // SAH, surface area heuristic calculation
    float minCost =
        work.size() * (bb.getXSize() * bb.getYSize() + bb.getYSize() * bb.getZSize() + bb.getZSize() * bb.getXSize());

    auto bestSplit = numeric_limits<typename BaseVecT::CoordType>::max();
    int bestAxis = -1;

    // try all 3 axises X = 0, Y = 1, Z = 2
    for (uint8_t j = 0; j < 3; j++)
    {
        auto axis = j;
        float start, stop, step;

        if (axis == 0)
        {
            start = bb.getMin().x;
            stop = bb.getMax().x;
        }
        else if(axis == 1)
        {
            start = bb.getMin().y;
            stop = bb.getMax().y;
        }
        else
        {
            start = bb.getMin().z;
            stop = bb.getMax().z;
        }

        if (fabs(stop - start) < 1e-4)
        {
            // bb side along this axis too short, we must move to a different axis
            continue;
        }

        step = (stop - start) / (1024.0f / (depth + 1.0f));

        for (float testSplit = start + step; testSplit < stop - step; testSplit += step)
        {
            // create left and right bb
            BoundingBox<BaseVecT> lBb;
            BoundingBox<BaseVecT> rBb;

            int countLeft = 0, countRight = 0;

            for (size_t i = 0; i < work.size(); i++)
            {
                auto& v = work[i];

                float value;
                if (axis == 0)
                {
                    value = v.bb.getCentroid().x;
                }
                else if (axis == 1)
                {
                    value = v.bb.getCentroid().y;
                }
                else
                {
                    value = v.bb.getCentroid().z;
                }

                if (value < testSplit)
                {
                    lBb.expand(v.bb);
                    countLeft++;
                }
                else
                {
                    rBb.expand(v.bb);
                    countRight++;
                }
            }

            if (countLeft <= 1 || countRight <= 1)
            {
                continue;
            }

            float surfaceLeft =
                lBb.getXSize() * lBb.getYSize() + lBb.getYSize() * lBb.getZSize() + lBb.getZSize() * lBb.getXSize();
            float surfaceRight =
                lBb.getXSize() * rBb.getYSize() + rBb.getYSize() * rBb.getZSize() + rBb.getZSize() * rBb.getXSize();

            float totalCost = surfaceLeft * countLeft + surfaceRight * countRight;

            if (totalCost < minCost)
            {
                minCost = totalCost;
                bestSplit = testSplit;
                bestAxis = axis;
            }
        }
    }

    if (bestAxis == -1)
    {
        auto leaf = make_unique<BVHLeaf<BaseVecT>>();
        for (auto aabb: work)
        {
            leaf->triangles.push_back(aabb.triangles.front());
        }
        return leaf;
    }

    vector<AABB_t> leftWork;
    vector<AABB_t> rightWork;
    BoundingBox<BaseVecT> lBb;
    BoundingBox<BaseVecT> rBb;

    for (size_t i = 0; i < work.size(); i++)
    {
        auto& v = work[i];

        float value;
        if (bestAxis == 0)
        {
            value = v.bb.getCentroid().x;
        }
        else if (bestAxis == 1)
        {
            value = v.bb.getCentroid().y;
        }
        else
        {
            value = v.bb.getCentroid().z;
        }

        if (value < bestSplit)
        {
            leftWork.push_back(v);
            lBb.expand(v.bb);
        }
        else
        {
            rightWork.push_back(v);
            rBb.expand(v.bb);
        }
    }

    auto inner = make_unique<BVHInner<BaseVecT>>();
    inner->left = buildTreeRecursive(leftWork, depth + 1);
    inner->left->bb = lBb;

    inner->right = buildTreeRecursive(rightWork, depth + 1);
    inner->right->bb = rBb;

    return inner;
}


}
