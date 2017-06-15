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
 * FinalizeAlgorithm.tcc
 *
 *  @date 13.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <lvr2/geometry/Normal.hpp>
#include <lvr2/util/VectorMap.hpp>

namespace lvr2
{

template<typename BaseVecT>
boost::shared_ptr<lvr::MeshBuffer> FinalizeAlgorithm<BaseVecT>::apply(const BaseMesh <BaseVecT>& mesh)
{
    // Create vertex and normal buffer
    std::vector<float> vertices;
    vertices.reserve(mesh.numVertices() * 3);
    std::vector<float> normals;
    normals.reserve(mesh.numVertices() * 3);
    VertexMap<size_t> idxMap;

    // TODO: use real normal
    Normal<BaseVecT> normal(0, 1, 0);
    for (size_t i = 0; i < mesh.numVertices(); i++)
    {
        // TODO: Don't create handle by yourself! This is extremly unsafe...
        VertexHandle handle(i);
        auto point = mesh.getPoint(handle);

        vertices.push_back(point.x);
        vertices.push_back(point.y);
        vertices.push_back(point.z);

        normals.push_back(normal.getX());
        normals.push_back(normal.getY());
        normals.push_back(normal.getZ());

        // Save index of vertex for face mapping
        idxMap.insert(handle, i);
    }

    // Create face buffer
    std::vector<unsigned int> faces;
    faces.reserve(mesh.numFaces() * 3);
    for (size_t i = 0; i < mesh.numFaces(); i++)
    {
        // TODO: Don't create handle by yourself! This is extremly unsafe...
        auto handles = mesh.getVertexHandlesOfFace(FaceHandle(i));
        for (auto handle: handles)
        {
            faces.push_back(idxMap[handle]);
        }
    }

    auto buffer = boost::make_shared<lvr::MeshBuffer>();
    buffer->setVertexArray(vertices);
    buffer->setVertexNormalArray(normals);
    buffer->setFaceArray(faces);

    return buffer;
}

} // namespace lvr2
