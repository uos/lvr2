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
 * Normals.hpp
 *
 *  @date 17.06.2017
 */

#include <vector>

using std::vector;

#include <lvr2/geometry/Normal.hpp>
#include <lvr2/util/VectorMap.hpp>

namespace lvr2
{

template <typename BaseVecT>
FaceMap<Normal<BaseVecT>> calcFaceNormals(const BaseMesh<BaseVecT>& mesh)
{
    FaceMap<Normal<BaseVecT>> out;
    for (auto faceH: mesh.faces())
    {
        auto vertexPositions = mesh.getVertexPositionsOfFace(faceH);

        auto v1 = vertexPositions[0];
        auto v2 = vertexPositions[1];
        auto v3 = vertexPositions[2];
        auto normal = (v1 - v2).cross(v1 - v3);
        out.insert(faceH, Normal<BaseVecT>(normal));
    }
    return out;
}

template <typename BaseVecT>
optional<Normal<BaseVecT>> interpolatedVertexNormal(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals,
    VertexHandle handle
)
{
    auto faces = mesh.getFacesOfVertex(handle);

    // Return none, if vertex does not have connected faces
    if (faces.empty())
    {
        return boost::none;
    }

    // Average normal over all connected faces
    Vector<BaseVecT> v(0, 0, 0);
    for (auto face: faces)
    {
        v += normals[face].asVector();
    }

    return v.normalized();
}

} // namespace lvr2
