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

namespace lvr2
{

template<typename BaseVecT>
BVHTree<BaseVecT>::BVHTree(const vector<float>& vertices, const vector<uint32_t>& faces)
{
    m_root = buildTree(vertices, faces);
}

template<typename BaseVecT>
BVHInner<BaseVecT> BVHTree<BaseVecT>::buildTree(const vector<float>& vertices, const vector<uint32_t>& faces)
{
    vector<BoundingBox<BaseVecT>> work;
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

        outerBb.expand(faceBb);
        work.push_back(faceBb);
    }

    std::cout << outerBb << " count: " << work.size() << std::endl;

    return BVHInner<BaseVecT>();
}

}
