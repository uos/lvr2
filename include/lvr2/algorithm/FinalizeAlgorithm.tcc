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
#include <cmath>

#include <lvr/io/Progress.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/attrmaps/AttrMaps.hpp>

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
void ClusterFlatteningFinalizer<BaseVecT>::setVertexColors(const VertexMap<Rgb8Color>& vertexColors)
{
    m_vertexColors = vertexColors;
}

template<typename BaseVecT>
void ClusterFlatteningFinalizer<BaseVecT>::setMaterializerResult(const MaterializerResult<BaseVecT>& matResult)
{
    m_materializerResult = matResult;
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

    // Create buffer and variables for texturizing
    bool useTextures = false;
    if (m_materializerResult && m_materializerResult.get().m_textures)
    {
        useTextures = true;
    }
    vector<float> texCoords;
    vector<lvr::Material*> materials;
    vector<unsigned int> faceMaterials;
    vector<GlTexture*> textures;

    // Global material index will be used for indexing materials in the faceMaterialIndexBuffer
    // The basic material will have the index 0
    unsigned int globalMaterialIndex = 1;
    // Create default material
    size_t defaultR = 0, defaultG = 0, defaultB = 0;
    lvr::Material* m = new lvr::Material;
    m->r = defaultR;
    m->g = defaultG;
    m->b = defaultB;
    m->texture_index = -1;
    materials.push_back(m);
    // This map remembers which texture and material are associated with each other
    std::map<int, unsigned int> textureMaterialMap; // Stores the ID of the material for each textureIndex
    textureMaterialMap[-1] = 0; // texIndex -1 => no texture => default material with index 0

    std::map<Rgb8Color, int> colorMaterialMap;

    // This map remembers which vertices were already visited for materials and textures
    // Each vertex must be visited exactly once
    SparseVertexMap<size_t> vertexVisitedMap;
    size_t vertexVisitCount = 0;

    // Create face buffer
    vector<unsigned int> faces;
    faces.reserve(mesh.numFaces() * 3);

    // This counter is used to determine the index of a newly inserted vertex
    size_t vertexCount = 0;

    // This map remembers which vertex we already inserted and at what
    // position. This is important to create the face map.
    SparseVertexMap<size_t> idxMap;

    string comment = lvr::timestamp.getElapsedTime() + "Finalizing mesh ";
    lvr::ProgressBar progress(m_cluster.numCluster(), comment);

    // Loop over all clusters
    for (auto clusterH: m_cluster)
    {
        idxMap.clear();
        vertexVisitedMap.clear();

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
        }

        // When using textures,
        // For each cluster:
        if (m_materializerResult)
        {
            // For each face in cluster:
            // Materials
            for (auto faceH : cluster.handles)
            {
                Material m = m_materializerResult.get().m_faceMaterials.get(faceH).get();

                // For each vertex in face:
                for (auto vertexH : mesh.getVerticesOfFace(faceH))
                {
                    if (!vertexVisitedMap.containsKey(vertexH))
                    {
                        bool vertexHasTexCoords = m_materializerResult.get()
                                .m_vertexTexCoords.get()
                                .get(vertexH);

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

                // For this face: generate materials
                bool faceHasTexture = m.m_texture; // optional
                bool faceHasColor = m.m_color; // optional

                // Does this face have a texture?
                if (useTextures && faceHasTexture)
                {
                    // Yes: read texture info
                    TextureHandle texHandle = m.m_texture.get();
                    auto texOptional = m_materializerResult.get()
                        .m_textures.get()
                        .get(texHandle);
                    const Texture<BaseVecT>& texture = texOptional.get();
                    int textureIndex = texture.m_index;

                    // Material for this texture already created?
                    if (textureMaterialMap.find(textureIndex) != textureMaterialMap.end())
                    {
                        // Yes: get index from map and push to faceMaterials buffer
                        unsigned int materialIndex = textureMaterialMap[textureIndex];
                        faceMaterials.push_back(materialIndex);
                    }
                    else
                    {
                        // No: create material with texture
                        lvr::Material* material = new lvr::Material;
                        material->r = defaultR;
                        material->g = defaultG;
                        material->b = defaultB;
                        material->texture_index = textureIndex;
                        materials.push_back(material);
                        faceMaterials.push_back(globalMaterialIndex);
                        textureMaterialMap[textureIndex] = globalMaterialIndex;
                        globalMaterialIndex++;
                    }
                }
                else if (faceHasColor)
                {
                    // Else: does this face have a color?
                    // Yes: read material(lvr2) and create buffer entry (lvr1 material)
                    Rgb8Color c = m.m_color.get();
                    if (colorMaterialMap.count(c))
                    {
                        faceMaterials.push_back(colorMaterialMap[c]);
                    }
                    else
                    {
                        colorMaterialMap[c] = globalMaterialIndex;
                        lvr::Material* material = new lvr::Material;
                        material->r = c[0];
                        material->g = c[1];
                        material->b = c[2];
                        material->texture_index = -1;
                        materials.push_back(material);
                        faceMaterials.push_back(globalMaterialIndex);
                        globalMaterialIndex++;
                    }
                }
                else
                {
                    // No texture, no colors: use default material
                    faceMaterials.push_back(0);
                }
            }
        }
    }

    cout << endl;


    auto buffer = boost::make_shared<lvr::MeshBuffer>();
    buffer->setVertexArray(vertices);
    buffer->setFaceArray(faces);

    if (m_vertexNormals)
    {
        buffer->setVertexNormalArray(normals);
    }

    if (m_clusterColors || m_vertexColors)
    {
        buffer->setVertexColorArray(colors);
    }

    if (m_materializerResult)
    {
        if (m_materializerResult.get().m_textures)
        {
            for (auto texH : m_materializerResult.get().m_textures.get())
            {
                // TODO: lvr2::texture konvertieren in lvr::GlTexture und ins array packen
            }
        }

        buffer->setMaterialArray(materials);
        buffer->setFaceMaterialIndexArray(faceMaterials);
        buffer->setVertexTextureCoordinateArray(texCoords);
        buffer->setTextureArray(textures); // TODO: siehe oben
    }

    return buffer;
}

} // namespace lvr2
