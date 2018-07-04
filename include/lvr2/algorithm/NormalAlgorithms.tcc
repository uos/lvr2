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
 * NormalAlgorithms.tcc
 *
 * @date 19.07.2017
 * @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <vector>

using std::vector;

namespace lvr2
{

template <typename BaseVecT>
boost::optional<Normal<BaseVecT>> getFaceNormal(array<Point<BaseVecT>, 3> vertices)
{
    auto v1 = vertices[0];
    auto v2 = vertices[1];
    auto v3 = vertices[2];
    auto normalDir = (v1 - v2).cross(v1 - v3);
    return normalDir.length2() == 0
        ? boost::none
        : boost::optional<Normal<BaseVecT>>(Normal<BaseVecT>(normalDir));
}

template <typename BaseVecT>
DenseFaceMap<Normal<BaseVecT>> calcFaceNormals(const BaseMesh<BaseVecT>& mesh)
{
    DenseFaceMap<Normal<BaseVecT>> out;
    out.reserve(mesh.numFaces());

    for (auto faceH: mesh.faces())
    {
        auto maybeNormal = getFaceNormal(mesh.getVertexPositionsOfFace(faceH));
        auto normal = maybeNormal
            ? *maybeNormal
            : Normal<BaseVecT>(0, 0, 1);
        out.insert(faceH, normal);
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

    // It is indeed possible that `v` is the zero vector here: if there are two
    // faces with normals pointing into exactly different directions.
    return v.length2() == 0 ? boost::none : boost::optional<Normal<BaseVecT>>(v.normalized());
}

template<typename BaseVecT>
DenseVertexMap<Normal<BaseVecT>> calcVertexNormals(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals,
    const PointsetSurface<BaseVecT>& surface
)
{
    DenseVertexMap<Normal<BaseVecT>> normalMap;
    normalMap.reserve(mesh.numVertices());

    for (auto vH: mesh.vertices())
    {
        // Use averaged normals from adjacent faces
        if (auto normal = interpolatedVertexNormal(mesh, normals, vH))
        {
            normalMap.insert(vH, *normal);
        }
        else
        {
            // Fall back to normals from point cloud
            if (!surface.pointBuffer()->hasNormals())
            {
                // The panic is justified here: in the process of creating the
                // mesh, normals have to be estimated. These normals are
                // written to the point buffer.
                panic("the point buffer needs normals!");
            }

            // Get idx for nearest point to vertex from point cloud
            auto vertex = mesh.getVertexPosition(vH);
            vector<size_t> pointIdx;
            surface.searchTree()->kSearch(vertex, 1, pointIdx);
            if (pointIdx.empty())
            {
                panic("no near point found!");
            }

            // Get normal for nearest vertex neighbour from point cloud
            if(auto normal = surface.pointBuffer()->getNormal(pointIdx[0]))
            {
                normalMap.insert(vH, *normal);
            }
            else
            {
                panic("no normal for point found!");
            }
        }
    }

    return normalMap;
}

template<typename BaseVecT>
DenseVertexMap<Normal<BaseVecT>> calcVertexNormals(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals
)
{
    DenseVertexMap<Normal<BaseVecT>> normalMap;
    normalMap.reserve(mesh.numVertices());

    for (auto vH: mesh.vertices())
    {
        // Use averaged normals from adjacent faces
        if (auto normal = interpolatedVertexNormal(mesh, normals, vH))
        {
            normalMap.insert(vH, *normal);
        }
        else
        {
            normalMap.insert(vH, Normal<BaseVecT>(0, 0, 1));
        }
    }

    return normalMap;
}

} // namespace lvr2
