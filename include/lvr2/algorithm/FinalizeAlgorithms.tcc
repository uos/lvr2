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
 * SimpleFinalizer.tcc
 *
 *  @date 13.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <vector>
#include <utility>
#include <cmath>

#include "lvr2/algorithm/Materializer.hpp"

#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/Progress.hpp"

#include "lvr2/util/Util.hpp"

namespace lvr2
{

template<typename BaseVecT>
MeshBufferPtr SimpleFinalizer<BaseVecT>::apply(const BaseMesh <BaseVecT>& mesh)
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

    // for all vertices
    size_t vertexCount = 0;
    for (auto vH : mesh.vertices())
    {
        auto point = mesh.getVertexPosition(vH);

        // add vertex positions to buffer
        vertices.push_back(point.x);
        vertices.push_back(point.y);
        vertices.push_back(point.z);

        if (m_normalData)
        {
            // add normal data to buffer if given
            auto normal = (*m_normalData)[vH];
            normals.push_back(normal.getX());
            normals.push_back(normal.getY());
            normals.push_back(normal.getZ());
        }

        if (m_colorData)
        {
            // add color data to buffer if given
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
            // add faces to buffer
            faces.push_back(idxMap[handle]);
        }
    }

    // create buffer object and pass values
    MeshBufferPtr buffer( new MeshBuffer );

    buffer->setVertices(Util::convert_vector_to_shared_array(vertices), vertices.size() / 3);
    buffer->setFaceIndices(Util::convert_vector_to_shared_array(faces), faces.size() / 3);

    if (m_normalData)
    {
        buffer->setVertexNormals(Util::convert_vector_to_shared_array(normals));
    }

    if (m_colorData)
    {
        buffer->setVertexColors(Util::convert_vector_to_shared_array(colors));
    }

    return buffer;
}

template<typename BaseVecT>
void SimpleFinalizer<BaseVecT>::setColorData(const VertexMap<Rgb8Color>& colorData)
{
    m_colorData = colorData;
}

template<typename BaseVecT>
void SimpleFinalizer<BaseVecT>::setNormalData(const VertexMap<Normal<typename BaseVecT::CoordType>>& normalData)
{
    m_normalData = normalData;
}

template<typename BaseVecT>
TextureFinalizer<BaseVecT>::TextureFinalizer(
    const ClusterBiMap<FaceHandle>& cluster
)
    : m_cluster(cluster)
{}

template<typename BaseVecT>
void TextureFinalizer<BaseVecT>::setVertexNormals(const VertexMap<Normal<typename BaseVecT::CoordType>>& normals)
{
    m_vertexNormals = normals;
}

template<typename BaseVecT>
void TextureFinalizer<BaseVecT>::setClusterColors(const ClusterMap<Rgb8Color>& colors)
{
    m_clusterColors = colors;
}

template<typename BaseVecT>
void TextureFinalizer<BaseVecT>::setVertexColors(const VertexMap<Rgb8Color>& vertexColors)
{
    m_vertexColors = vertexColors;
}

template<typename BaseVecT>
void TextureFinalizer<BaseVecT>::setMaterializerResult(const MaterializerResult<BaseVecT>& matResult)
{
    m_materializerResult = matResult;
}


template<typename BaseVecT>
MeshBufferPtr TextureFinalizer<BaseVecT>::apply(const BaseMesh<BaseVecT>& mesh)
{
    // Create vertex buffer and all buffers holding vertex attributes
    vector<float> vertices;
    vertices.reserve(mesh.numVertices() * 3 * 2);

    vector<float> normals;
    if (m_vertexNormals)
    {
        normals.reserve(mesh.numVertices() * 3 * 2);
    }

    vector<unsigned char> colors;
    if (m_clusterColors)
    {
        colors.reserve(mesh.numVertices() * 3 * 2);
    }

    // Create buffer and variables for texturizing
    bool useTextures = false;
    if (m_materializerResult && m_materializerResult.get().m_textures)
    {
        useTextures = true;
    }
    vector<float> texCoords;
    vector<Material> materials;
    vector<unsigned int> faceMaterials;
    vector<unsigned int> clusterMaterials;
    vector<Texture> textures;
    vector<vector<unsigned int>> clusterFaceIndices;
    size_t clusterCount = 0;
    size_t faceCount = 0;

    // Global material index will be used for indexing materials in the faceMaterialIndexBuffer
    // The basic material will have the index 0
    unsigned int globalMaterialIndex = 1;
    // Create default material
    unsigned char defaultR = 0, defaultG = 0, defaultB = 0;
    Material m;
    std::array<unsigned char, 3> arr = {defaultR, defaultG, defaultB};
    m.m_color = std::move(arr);
    materials.push_back(m);
    // This map remembers which texture and material are associated with each other
    std::map<int, unsigned int> textureMaterialMap; // Stores the ID of the material for each textureIndex
    textureMaterialMap[-1] = 0; // texIndex -1 => no texture => default material with index 0

    std::map<Rgb8Color, int> colorMaterialMap;

    // Create face buffer
    vector<unsigned int> faces;
    faces.reserve(mesh.numFaces() * 3);

    // This counter is used to determine the index of a newly inserted vertex
    size_t vertexCount = 0;

    string comment = timestamp.getElapsedTime() + "Finalizing mesh ";
    ProgressBar progress(m_cluster.numCluster(), comment);

    // Loop over all clusters
    for (auto clusterH: m_cluster)
    {
        // This map remembers which vertex we already inserted and at what
        // position. This is important to create the face map.
        SparseVertexMap<size_t> idxMap;

        // Vector for storing indices of created faces
        vector<unsigned int> faceIndices;

        ++progress;

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

                    // If individual vertex colors are present: use these
                    if (m_vertexColors)
                    {
                        colors.push_back(static_cast<unsigned char>((*m_vertexColors)[vertexH][0]));
                        colors.push_back(static_cast<unsigned char>((*m_vertexColors)[vertexH][1]));
                        colors.push_back(static_cast<unsigned char>((*m_vertexColors)[vertexH][2]));
                    }
                    else if (m_clusterColors)
                    {
                        // else: use cluster colors if present
                        colors.push_back(static_cast<unsigned char>((*m_clusterColors)[clusterH][0]));
                        colors.push_back(static_cast<unsigned char>((*m_clusterColors)[clusterH][1]));
                        colors.push_back(static_cast<unsigned char>((*m_clusterColors)[clusterH][2]));
                    } // else: no colors

                    // Save index of vertex for face mapping
                    idxMap.insert(vertexH, vertexCount);
                    vertexCount++;
                }

                // At this point we know that the vertex is certainly in the
                // map (and the buffers).
                faces.push_back(idxMap[vertexH]);
            }
            faceIndices.push_back(faceCount++);
        }

        clusterFaceIndices.push_back(faceIndices);

        // When using textures,
        // For each cluster:
        if (m_materializerResult)
        {
            // This map remembers which vertices were already visited for materials and textures
            // Each vertex must be visited exactly once
            SparseVertexMap<size_t> vertexVisitedMap;
            size_t vertexVisitCount = 0;

            Material m = m_materializerResult.get().m_clusterMaterials.get(clusterH).get();
            bool clusterHasTextures = static_cast<bool>(m.m_texture); // optional
            bool clusterHasColor = static_cast<bool>(m.m_color); // optional

            unsigned int materialIndex;

            // Does this cluster use textures?
            if (useTextures && clusterHasTextures)
            {
                // Yes: read texture info
                TextureHandle texHandle = m.m_texture.get();
                auto texOptional = m_materializerResult.get()
                    .m_textures.get()
                    .get(texHandle);
                const Texture& texture = texOptional.get();
                int textureIndex = texture.m_index;

                // Material for this texture already created?
                if (textureMaterialMap.find(textureIndex) != textureMaterialMap.end())
                {
                    // Yes: get index from map
                    materialIndex = textureMaterialMap[textureIndex];
                }
                else
                {
                    // No: create material with texture
                    materials.push_back(m);
                    textures.push_back(texture);
                    textureMaterialMap[textureIndex] = globalMaterialIndex;
                    materialIndex = globalMaterialIndex;
                    globalMaterialIndex++;
                }
            }
            else if (clusterHasColor)
            {
                // Else: does this face have a color?
                Rgb8Color c = m.m_color.get();
                if (colorMaterialMap.count(c))
                {
                    materialIndex = colorMaterialMap[c];
                }
                else
                {
                    colorMaterialMap[c] = globalMaterialIndex;
                    materials.push_back(m);
                    materialIndex = globalMaterialIndex;
                    globalMaterialIndex++;
                }
            }
            else
            {
                materialIndex = 0;
            }

            clusterMaterials.push_back(materialIndex);



            // For each face in cluster:
            // Materials
            for (auto faceH : cluster.handles)
            {

                faceMaterials.push_back(materialIndex);

                // For each vertex in face:
                for (auto vertexH : mesh.getVerticesOfFace(faceH))
                {
                    if (!vertexVisitedMap.containsKey(vertexH))
                    {
                        auto& vertexTexCoords = m_materializerResult.get().m_vertexTexCoords;
                        bool vertexHasTexCoords = vertexTexCoords.is_initialized()
                                                  ? static_cast<bool>(vertexTexCoords.get().get(vertexH))
                                                  : false;

                        if (useTextures && vertexHasTexCoords)
                        {
                            // Use tex coord vertex map to find texture coords
                            const TexCoords coords = m_materializerResult.get()
                                .m_vertexTexCoords.get()
                                .get(vertexH).get()
                                .getTexCoords(clusterH);

                            texCoords.push_back(coords.u);
                            texCoords.push_back(coords.v);
                            texCoords.push_back(0.0);
                        } else {
                            // Cluster does not have a texture, use default coords
                            // Every vertex needs an entry in this buffer,
                            // This is why 0's are inserted
                            texCoords.push_back(0.0);
                            texCoords.push_back(0.0);
                            texCoords.push_back(0.0);
                        }
                        vertexVisitedMap.insert(vertexH, vertexVisitCount);
                        vertexVisitCount++;
                    }
                }
            }
        }
    }

    cout << endl;

    MeshBufferPtr buffer = MeshBufferPtr( new MeshBuffer );
    buffer->setVertices(Util::convert_vector_to_shared_array(vertices), vertices.size() / 3);
    buffer->setFaceIndices(Util::convert_vector_to_shared_array(faces), faces.size() / 3);

    if (m_vertexNormals)
    {
        buffer->setVertexNormals(Util::convert_vector_to_shared_array(normals));
    }

    if (m_clusterColors || m_vertexColors)
    {
        buffer->setVertexColors(Util::convert_vector_to_shared_array(colors));
    }

    if (m_materializerResult)
    {
        vector<Material> &mats = buffer->getMaterials();
        vector<Texture> &texts = buffer->getTextures();
        mats.insert(mats.end(), materials.begin(), materials.end());
        texts.insert(texts.end(), textures.begin(), textures.end());

        buffer->setFaceMaterialIndices(Util::convert_vector_to_shared_array(faceMaterials));
        buffer->addIndexChannel(Util::convert_vector_to_shared_array(clusterMaterials), "cluster_material_indices", clusterMaterials.size(), 1);

        // convert from 3 floats per vertex to 2...
        floatArr texCoordsArr = floatArr( new float[(texCoords.size() / 3) * 2] );

        for (size_t i = 0; i < texCoords.size() / 3; i++)
        {
            texCoordsArr[i*2 + 0] = texCoords[i*3 + 0];
            texCoordsArr[i*2 + 1] = texCoords[i*3 + 1];
        }
        buffer->setTextureCoordinates(texCoordsArr);

        // TODO TALK TO THOMAS
        for (size_t i = 0; i < clusterFaceIndices.size(); i++)
        {
            std::string cluster_name = "cluster" + std::to_string(i) + "_face_indices";
            buffer->addIndexChannel(Util::convert_vector_to_shared_array(clusterFaceIndices[i]), cluster_name, clusterFaceIndices[i].size(), 1);
        }
    }

    return buffer;
}

} // namespace lvr2
