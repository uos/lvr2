/**
 * Copyright (c) 2019, University Osnabrück
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

/**
 * ChunkBuilder.cpp
 *
 * @date 21.07.2019
 * @author Malte kl. Piening
 * @author Raphael Marx
 */

#include "lvr2/algorithm/ChunkBuilder.hpp"

namespace lvr2
{

ChunkBuilder::ChunkBuilder(
    std::shared_ptr<HalfEdgeMesh<BaseVector<float>>> originalMesh,
    std::shared_ptr<std::unordered_map<unsigned int, std::vector<std::weak_ptr<ChunkBuilder>>>>
        vertexUse)
    : m_originalMesh(originalMesh), m_vertexUse(vertexUse)
{
}

ChunkBuilder::~ChunkBuilder() {}

void ChunkBuilder::addFace(const FaceHandle& faceHandle)
{
    // add the original index of the face to the chunk with its chunkID
    m_faces.push_back(faceHandle);

    std::array<VertexHandle, 3> vertices = m_originalMesh->getVerticesOfFace(faceHandle);

    // next add the vertices of the face to the chunk
    for (unsigned int i = 0; i < 3; i++)
    {
        std::weak_ptr<ChunkBuilder> thisBuilderPtr(shared_from_this());
        // if the vertex is not in the vector, we add the vertex (we just mark the vertex by adding
        // the chunkID)
        if (m_vertexUse->find(vertices[i].idx()) == m_vertexUse->end())
        {
            m_vertexUse->insert({vertices[i].idx(), std::vector<std::weak_ptr<ChunkBuilder>>()});
        }
        if (std::find_if(m_vertexUse->at(vertices[i].idx()).begin(),
                         m_vertexUse->at(vertices[i].idx()).end(),
                         [&thisBuilderPtr](const std::weak_ptr<ChunkBuilder>& otherBuilderPtr) {
                             return thisBuilderPtr.lock() == otherBuilderPtr.lock();
                         })
            == m_vertexUse->at(vertices[i].idx()).end())
        {
            m_vertexUse->at(vertices[i].idx()).push_back(shared_from_this());

            m_numVertices++;

            if (m_vertexUse->at(vertices[i].idx()).size() == 2)
            {
                for (unsigned int j = 0; j < 2; j++)
                {
                    m_vertexUse->at(vertices[i].idx())[j].lock()->addDuplicateVertex(vertices[i]);
                }
            }
            else if (m_vertexUse->at(vertices[i].idx()).size() > 2)
            {
                m_vertexUse->at(vertices[i].idx()).back().lock()->addDuplicateVertex(vertices[i]);
            }
        }
    }
}

void ChunkBuilder::addDuplicateVertex(const VertexHandle& index)
{
    if (std::find(m_duplicateVertices.begin(), m_duplicateVertices.end(), index)
        == m_duplicateVertices.end())
    {
        m_duplicateVertices.push_back(index);
    }
}

unsigned int ChunkBuilder::numFaces() const
{
    return m_faces.size();
}

unsigned int ChunkBuilder::numVertices() const
{
    return m_numVertices;
}

MeshBufferPtr ChunkBuilder::buildMesh(
    MeshBufferPtr attributedMesh,
    std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitVertices,
    std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitFaces) const
{
    std::unordered_map<unsigned int, unsigned int> vertexIndices;

    lvr2::floatArr vertices(new float[numVertices() * 3]);
    lvr2::indexArray faceIndices(new unsigned int[numFaces() * 3]);

    // build new model by adding vertices, faces and attribute channels
    lvr2::MeshBufferPtr mesh(new lvr2::MeshBuffer);

    // TODO: add more types if needed; it is more efficient to sort the channel in the channel manager
    std::vector<std::string> vertexChannelUChar;
    std::vector<std::string> vertexChannelUInt;
    std::vector<std::string> vertexChannelFloat;
    std::vector<std::string> faceChannelUChar;
    std::vector<std::string> faceChannelUInt;
    std::vector<std::string> faceChannelFloat;


    for (auto elem : *attributedMesh)
    {
        if(elem.first != "vertices" && elem.first != "face_indices")
        {
            if(elem.second.is_type<unsigned char>())
            {
                if(elem.second.numElements() == attributedMesh->numVertices())
                {
                    mesh->addEmptyUCharChannel(elem.first, numVertices(), elem.second.width());
                    vertexChannelUChar.push_back(elem.first);
                }
                else if(elem.second.numElements() == attributedMesh->numFaces())
                {
                    mesh->addEmptyUCharChannel(elem.first, numFaces(), elem.second.width());
                    faceChannelUChar.push_back(elem.first);
                }
                else
                {
                    // a channel that is not a vertex or a face channel will be added unchanged to each chunk
                    mesh->addUCharChannel(std::make_shared<
                            lvr2::Channel<unsigned char>>(attributedMesh->getUCharChannel(elem.first).get()), elem.first);
                }
            }
            else if(elem.second.is_type<unsigned int>())
            {
                if(elem.second.numElements() == attributedMesh->numVertices())
                {
                    mesh->addEmptyIndexChannel(elem.first, numVertices(), elem.second.width());
                    vertexChannelUInt.push_back(elem.first);
                }
                else if(elem.second.numElements() == attributedMesh->numFaces())
                {
                    mesh->addEmptyIndexChannel(elem.first, numFaces(), elem.second.width());
                    faceChannelUInt.push_back(elem.first);
                }
                else
                {
                    // a channel that is not a vertex or a face channel will be added unchanged to each chunk
                    mesh->addIndexChannel(std::make_shared<
                            lvr2::Channel<unsigned int>>(attributedMesh->getIndexChannel(elem.first).get()), elem.first);
                }
            }
            else if(elem.second.is_type<float>())
            {
                if(elem.second.numElements() == attributedMesh->numVertices())
                {
                    mesh->addEmptyFloatChannel(elem.first, numVertices(), elem.second.width());
                    vertexChannelFloat.push_back(elem.first);
                }
                else if(elem.second.numElements() == attributedMesh->numFaces())
                {
                    mesh->addEmptyFloatChannel(elem.first, numFaces(), elem.second.width());
                    faceChannelFloat.push_back(elem.first);
                }
                else
                {
                    // a channel that is not a vertex or a face channel will be added unchanged to each chunk
                    mesh->addFloatChannel(std::make_shared<
                            lvr2::Channel<float>>(attributedMesh->getFloatChannel(elem.first).get()), elem.first);
                }
            }
            else
            {

            }

        }

    }


    // fill vertex buffer with duplicate vertices
    for (VertexHandle vertex : m_duplicateVertices)
    {
        if (vertexIndices.find(vertex.idx()) == vertexIndices.end())
        {
            unsigned int vertexIndex                               = vertexIndices.size();
            vertexIndices[static_cast<unsigned int>(vertex.idx())] = vertexIndex;

            // apply vertex position
            for (uint8_t j = 0; j < 3; j++)
            {
                vertices[vertexIndex * 3 + j] = m_originalMesh->getVertexPosition(vertex)[j];
            }

            // apply vertex attributes
            unsigned int attributedVertexIndex = vertex.idx();
            if (splitVertices->find(attributedVertexIndex) != splitVertices->end())
            {
                attributedVertexIndex = splitVertices->at(attributedVertexIndex);
            }


            // vertex channels
            for(const std::string& name : vertexChannelUChar)
            {
                Channel<unsigned char> tmp = mesh->getUCharChannel(name).get();
                for(size_t component = 0; component < tmp.width(); component++)
                {

                    tmp.dataPtr()[vertexIndex * tmp.width() + component]
                            = attributedMesh->getUCharHandle(attributedVertexIndex, name)[component];

//                    // alternative way using Handles instead of dataPtr():
//                    mesh->getUCharHandle(vertexIndex, name)[component]
//                            = attributedMesh->getFloatHandle(attributedVertexIndex, name)[component];
                }
            }
            for(const std::string& name : vertexChannelUInt)
            {
                Channel<unsigned int> tmp = mesh->getIndexChannel(name).get();
                for(size_t component = 0; component < tmp.width(); component++)
                {
                    tmp.dataPtr()[vertexIndex * tmp.width() + component]
                            = attributedMesh->getIndexHandle(attributedVertexIndex, name)[component];
                }
            }
            for(const std::string& name : vertexChannelFloat)
            {
                Channel<float> tmp = mesh->getFloatChannel(name).get();

                for(size_t component = 0; component < tmp.width(); component++)
                {
                    tmp.dataPtr()[vertexIndex * tmp.width() + component]
                            = attributedMesh->getFloatHandle(attributedVertexIndex, name)[component];
                }
            }
        }
    }

    // fill new vertex and face buffer
    for (unsigned int face = 0; face < m_faces.size(); face++)
    {
        for (uint8_t faceVertex = 0; faceVertex < 3; faceVertex++)
        {
            if (vertexIndices.find(
                    m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex].idx())
                == vertexIndices.end())
            {
                unsigned int vertexIndex = vertexIndices.size();
                vertexIndices[static_cast<unsigned int>(
                    m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex].idx())]
                    = vertexIndex;

                // apply vertex position
                for (uint8_t vertexComponent = 0; vertexComponent < 3; vertexComponent++)
                {
                    vertices[vertexIndex * 3 + vertexComponent]
                        = m_originalMesh->getVertexPosition(m_originalMesh->getVerticesOfFace(
                            m_faces[face])[faceVertex])[vertexComponent];
                }

                // apply vertex attributes
                unsigned int attributedVertexIndex
                    = m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex].idx();
                if (splitVertices->find(attributedVertexIndex) != splitVertices->end())
                {
                    attributedVertexIndex = splitVertices->at(attributedVertexIndex);
                }

                // vertex channels
                for(const std::string& name : vertexChannelUChar)
                {
                    Channel<unsigned char> tmp = mesh->getUCharChannel(name).get();
                    for(size_t component = 0; component < tmp.width(); component++)
                    {
                        tmp.dataPtr()[vertexIndex * tmp.width() + component]
                                = attributedMesh->getUCharHandle(attributedVertexIndex, name)[component];

//                        // alternative way using Handles instead of dataPtr():
//                        mesh->getUCharHandle(vertexIndex, name)[component]
//                              = attributedMesh->getFloatHandle(attributedVertexIndex, name)[component];
                    }
                }
                for(const std::string& name : vertexChannelUInt)
                {
                    Channel<unsigned int> tmp = mesh->getIndexChannel(name).get();
                    for(size_t component = 0; component < tmp.width(); component++)
                    {
                        tmp.dataPtr()[vertexIndex * tmp.width() + component]
                                = attributedMesh->getIndexHandle(attributedVertexIndex, name)[component];
                    }
                }
                for(const std::string& name : vertexChannelFloat)
                {
                    Channel<float> tmp = mesh->getFloatChannel(name).get();

                    for(size_t component = 0; component < tmp.width(); component++)
                    {
                        mesh->getFloatHandle(vertexIndex, name)[component]
                                = attributedMesh->getFloatHandle(attributedVertexIndex, name)[component];
                    }
                }
            }

            // apply face vertex
            faceIndices[face * 3 + faceVertex] = vertexIndices[static_cast<unsigned int>(
                m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex].idx())];
        }

        // apply face attributes
        unsigned int attributedFaceIndex = m_faces[face].idx();
        if (splitVertices->find(attributedFaceIndex) != splitVertices->end())
        {
            attributedFaceIndex = splitVertices->at(attributedFaceIndex);
        }

        // face channels
        for(const std::string& name : faceChannelUChar)
        {
            Channel<unsigned char> tmp = mesh->getUCharChannel(name).get();
            for(size_t component = 0; component < tmp.width(); component++)
            {
                tmp.dataPtr()[face * tmp.width() + component]
                        = attributedMesh->getUCharHandle(attributedFaceIndex, name)[component];

//                        // alternative way using Handles instead of dataPtr():
//                        mesh->getUCharHandle(face, name)[component]
//                              = attributedMesh->getFloatHandle(attibutedFaceIndex, name)[component];
            }
        }
        for(const std::string& name : faceChannelUInt)
        {
            Channel<unsigned int> tmp = mesh->getIndexChannel(name).get();
            for(size_t component = 0; component < tmp.width(); component++)
            {
                tmp.dataPtr()[face * tmp.width() + component]
                        = attributedMesh->getIndexHandle(attributedFaceIndex, name)[component];
            }
        }
        for(const std::string& name : faceChannelFloat)
        {
            Channel<float> tmp = mesh->getFloatChannel(name).get();

            for(size_t component = 0; component < tmp.width(); component++)
            {
                mesh->getFloatHandle(face, name)[component]
                        = attributedMesh->getFloatHandle(attributedFaceIndex, name)[component];
            }
        }
    }

    // add vertices and face_indices to the mesh
    mesh->setVertices(vertices, numVertices());
    mesh->setFaceIndices(faceIndices, numFaces());

    mesh->addAtomic<unsigned int>(m_duplicateVertices.size(), "num_duplicates");

    return mesh;
}

} // namespace lvr2
