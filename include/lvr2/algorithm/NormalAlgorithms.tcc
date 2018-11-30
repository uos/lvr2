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
boost::optional<Normal<BaseVecT>> getFaceNormal(array<Vector<BaseVecT>, 3> vertices)
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
        v += normals[face];
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
            FloatChannelOptional normals = surface.pointBuffer()->getFloatChannel("normals");
            if(normals)
            {
                Normal<BaseVecT> normal = (*normals)[pointIdx[0]];
                normalMap.insert(vH, normal);
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
