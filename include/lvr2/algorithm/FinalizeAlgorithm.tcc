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
    VertexMap<size_t> idxMap;
    std::vector<float> vertices;
    vertices.reserve(mesh.numVertices() * 3);

    std::vector<float> normals;
    if (m_normalData)
    {
        normals.reserve(mesh.numVertices() * 3);
    }

    std::vector<unsigned char> colors;
    if (m_colorData)
    {
        colors.reserve(mesh.numVertices() * 3);
    }

    size_t vertexCount = 0;
    for (auto vH : mesh.vertices())
    {
        auto point = mesh.getVertexPosition(vH);

        vertices.push_back(point.x);
        vertices.push_back(point.y);
        vertices.push_back(point.z);

        if (m_normalData)
        {
            auto normal = (*m_normalData)[vH];
            normals.push_back(normal.getX());
            normals.push_back(normal.getY());
            normals.push_back(normal.getZ());
        }

        if (m_colorData)
        {
            colors.push_back(static_cast<unsigned char>((*m_colorData)[vH][0]));
            colors.push_back(static_cast<unsigned char>((*m_colorData)[vH][1]));
            colors.push_back(static_cast<unsigned char>((*m_colorData)[vH][2]));
        }

        // Save index of vertex for face mapping
        idxMap.insert(vH, vertexCount);
        vertexCount++;
    }

    // Create face buffer
    std::vector<unsigned int> faces;
    faces.reserve(mesh.numFaces() * 3);
    for (auto fH : mesh.faces())
    {
        auto handles = mesh.getVertexHandlesOfFace(fH);
        for (auto handle : handles)
        {
            faces.push_back(idxMap[handle]);
        }
    }

    auto buffer = boost::make_shared<lvr::MeshBuffer>();
    buffer->setVertexArray(vertices);
    buffer->setFaceArray(faces);

    if (m_normalData)
    {
        buffer->setVertexNormalArray(normals);
    }

    if (m_colorData)
    {
        buffer->setVertexColorArray(colors);
    }

    return buffer;
}

template<typename BaseVecT>
void FinalizeAlgorithm<BaseVecT>::setColorData(const VertexMap<ClusterPainter::Rgb8Color>& colorData)
{
    m_colorData = colorData;
}

template<typename BaseVecT>
void FinalizeAlgorithm<BaseVecT>::setNormalData(const VertexMap<Normal<BaseVecT>>& normalData)
{
    m_normalData = normalData;
}

} // namespace lvr2
