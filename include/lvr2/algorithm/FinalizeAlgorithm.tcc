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

#include <vector>
#include <utility>

#include <lvr2/geometry/Normal.hpp>
#include <lvr2/attrmaps/VectorMap.hpp>

namespace lvr2
{

template<typename BaseVecT>
boost::shared_ptr<lvr::MeshBuffer> FinalizeAlgorithm<BaseVecT>::apply(const BaseMesh <BaseVecT>& mesh)
{
    // Create vertex and normal buffer
    DenseVertexMap<size_t> idxMap;
    idxMap.reserve(mesh.numVertices());

    vector<float> vertices;
    vertices.reserve(mesh.numVertices() * 3);

    vector<float> normals;
    if (m_normalData)
    {
        normals.reserve(mesh.numVertices() * 3);
    }

    vector<unsigned char> colors;
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
    vector<unsigned int> faces;
    faces.reserve(mesh.numFaces() * 3);
    for (auto fH : mesh.faces())
    {
        auto handles = mesh.getVerticesOfFace(fH);
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
void FinalizeAlgorithm<BaseVecT>::setColorData(const VertexMap<Rgb8Color>& colorData)
{
    m_colorData = colorData;
}

template<typename BaseVecT>
void FinalizeAlgorithm<BaseVecT>::setNormalData(const VertexMap<Normal<BaseVecT>>& normalData)
{
    m_normalData = normalData;
}

template<typename BaseVecT>
ClusterFlatteningFinalizer<BaseVecT>::ClusterFlatteningFinalizer(
    const ClusterBiMap<FaceHandle>& cluster
)
    : m_cluster(cluster)
{}

template<typename BaseVecT>
void ClusterFlatteningFinalizer<BaseVecT>::setVertexNormals(const VertexMap<Normal<BaseVecT>>& normals)
{
    m_vertexNormals = normals;
}

template<typename BaseVecT>
void ClusterFlatteningFinalizer<BaseVecT>::setClusterColors(const ClusterMap<Rgb8Color>& colors)
{
    m_clusterColors = colors;
}

template<typename BaseVecT>
boost::shared_ptr<lvr::MeshBuffer>
    ClusterFlatteningFinalizer<BaseVecT>::apply(const BaseMesh<BaseVecT>& mesh)
{
    // Create vertex buffer and all buffers holding vertex attributes
    vector<float> vertices;
    vertices.reserve(mesh.numVertices() * 3);

    vector<float> normals;
    if (m_vertexNormals)
    {
        normals.reserve(mesh.numVertices() * 3);
    }

    vector<unsigned char> colors;
    if (m_clusterColors)
    {
        colors.reserve(mesh.numVertices() * 3);
    }

    // Create face buffer
    vector<unsigned int> faces;
    faces.reserve(mesh.numFaces() * 3);

    // This counter is used to determine the index of a newly inserted vertex
    size_t vertexCount = 0;

    // This map remembers which vertex we already inserted and at what
    // position. This is important to create the face map.
    SparseVertexMap<size_t> idxMap;

    // Loop over all clusters
    for (auto clusterH: m_cluster)
    {
        idxMap.clear();
        auto& cluster = m_cluster.getCluster(clusterH);

        // Loop over all faces of the cluster
        for (auto faceH: cluster.handles)
        {
            for (auto vertexH: mesh.getVerticesOfFace(faceH))
            {
                // Check if we already inserted this vertex. If not...
                if (!idxMap.containsKey(vertexH))
                {
                    // ... insert it into the buffers (with all its attributes)
                    auto point = mesh.getVertexPosition(vertexH);

                    vertices.push_back(point.x);
                    vertices.push_back(point.y);
                    vertices.push_back(point.z);

                    if (m_vertexNormals)
                    {
                        auto normal = (*m_vertexNormals)[vertexH];
                        normals.push_back(normal.getX());
                        normals.push_back(normal.getY());
                        normals.push_back(normal.getZ());
                    }

                    if (m_clusterColors)
                    {
                        colors.push_back(static_cast<unsigned char>((*m_clusterColors)[clusterH][0]));
                        colors.push_back(static_cast<unsigned char>((*m_clusterColors)[clusterH][1]));
                        colors.push_back(static_cast<unsigned char>((*m_clusterColors)[clusterH][2]));
                    }

                    // Save index of vertex for face mapping
                    idxMap.insert(vertexH, vertexCount);
                    vertexCount++;
                }

                // At this point we know that the vertex is certainly in the
                // map (and the buffers).
                faces.push_back(idxMap[vertexH]);
            }
        }
    }


    auto buffer = boost::make_shared<lvr::MeshBuffer>();
    buffer->setVertexArray(vertices);
    buffer->setFaceArray(faces);

    if (m_vertexNormals)
    {
        buffer->setVertexNormalArray(normals);
    }

    if (m_clusterColors)
    {
        buffer->setVertexColorArray(colors);
    }

    return buffer;
}


} // namespace lvr2
