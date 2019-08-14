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
 */

#include "lvr2/algorithm/ChunkBuilder.hpp"

namespace lvr2
{

ChunkBuilder::ChunkBuilder(
    std::shared_ptr<HalfEdgeMesh<BaseVector<float>>> originalMesh,
    std::shared_ptr<std::unordered_map<unsigned int, std::vector<std::weak_ptr<ChunkBuilder>>>> vertexUse)
    : m_originalMesh(originalMesh), m_vertexUse(vertexUse)
{
}

ChunkBuilder::~ChunkBuilder() {}

void ChunkBuilder::addFace(FaceHandle faceHandle)
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
            m_vertexUse->at(vertices[i].idx())
                .push_back(shared_from_this());

            m_numVertices++;

            if (m_vertexUse->at(vertices[i].idx()).size() == 2)
            {
                for (unsigned int j = 0; j < 2; j++)
                {
                    m_vertexUse->at(vertices[i].idx())[j]
                        .lock()
                        ->addDuplicateVertex(vertices[i]);
                }
            }
            else if (m_vertexUse->at(vertices[i].idx()).size() > 2)
            {
                m_vertexUse->at(vertices[i].idx())
                    .back()
                    .lock()
                    ->addDuplicateVertex(vertices[i]);
            }
        }
    }
}

void ChunkBuilder::addDuplicateVertex(VertexHandle index)
{
    if (std::find(m_duplicateVertices.begin(), m_duplicateVertices.end(), index)
        == m_duplicateVertices.end())
    {
        m_duplicateVertices.push_back(index);
    }
}

unsigned int ChunkBuilder::numFaces()
{
    return m_faces.size();
}

unsigned int ChunkBuilder::numVertices()
{
    return m_numVertices;
}

MeshBufferPtr ChunkBuilder::buildMesh()
{
    std::unordered_map<unsigned int, unsigned int> vertexIndices;

    lvr2::floatArr vertices(new float[numVertices() * 3]);
    lvr2::indexArray faceIndices(new unsigned int[numFaces() * 3]);

    lvr2::ucharArr faceColors;
    lvr2::ucharArr vertexColors;
    lvr2::floatArr faceNormals;
    lvr2::floatArr vertexNormals;

    /*if (m_originalMesh->hasFaceColors())
    {
        faceColors = lvr2::ucharArr(new unsigned char[numFaces() * 3]);
    }
    if (m_originalMesh->hasVertexColors())
    {
        vertexColors = lvr2::ucharArr(new unsigned char[numVertices() * 3]);
    }
    if (m_originalMesh->hasFaceNormals())
    {
        faceNormals = lvr2::floatArr(new float[numFaces() * 3]);
    }
    if (m_originalMesh->hasVertexNormals())
    {
        vertexNormals = lvr2::floatArr(new float[numVertices() * 3]);
    }*/

    // fill vertex buffer with duplicate vertices
    for (VertexHandle vertex : m_duplicateVertices)
    {
        if (vertexIndices.find(vertex.idx()) == vertexIndices.end())
        {
            unsigned int vertexIndex = vertexIndices.size();
            vertexIndices[(unsigned int) vertex.idx()]    = vertexIndex;

            for (uint8_t j = 0; j < 3; j++)
            {
                vertices[vertexIndex * 3 + j] = m_originalMesh->getVertexPosition(vertex)[j];

                /*if (m_originalMesh->hasVertexColors())
                {
                    size_t width = 3;
                    vertexColors[vertexIndex * 3 + j]
                        = m_originalMesh->getVertexColors(width)[vertex * 3 + j];
                }

                if (m_originalMesh->hasVertexNormals())
                {
                    vertexNormals[vertexIndex * 3 + j]
                        = m_originalMesh->getVertexNormals()[vertex * 3 + j];
                }*/
            }
        }
    }

    // fill new vertex and face buffer
    for (unsigned int face = 0; face < m_faces.size(); face++)
    {
        for (uint8_t faceVertex = 0; faceVertex < 3; faceVertex++)
        {
            if (vertexIndices.find(m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex].idx())
                == vertexIndices.end())
            {
                unsigned int vertexIndex = vertexIndices.size();
                vertexIndices[(unsigned int) m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex].idx()]
                    = vertexIndex;

                for (uint8_t vertexComponent = 0; vertexComponent < 3; vertexComponent++)
                {
                    vertices[vertexIndex * 3 + vertexComponent]
                        = m_originalMesh->getVertexPosition(m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex])[vertexComponent];

                    /*if (m_originalMesh->hasVertexColors())
                    {
                        size_t width = 3;
                        vertexColors[vertexIndex * 3 + vertexComponent]
                            = m_originalMesh->getVertexColors(
                                width)[m_originalMesh
                                               ->getFaceIndices()[m_faces[face] * 3 + faceVertex]
                                           * 3
                                       + vertexComponent];
                    }
                    if (m_originalMesh->hasVertexNormals())
                    {
                        vertexNormals[vertexIndex * 3 + vertexComponent]
                            = m_originalMesh->getVertexNormals()
                                  [m_originalMesh->getFaceIndices()[m_faces[face] * 3 + faceVertex]
                                       * 3
                                   + vertexComponent];
                    }*/
                }
            }

            faceIndices[face * 3 + faceVertex]
                = vertexIndices[(unsigned int) m_originalMesh->getVerticesOfFace(m_faces[face])[faceVertex].idx()];

            /*if (m_originalMesh->hasFaceColors())
            {
                size_t width = 3;
                faceColors[face * 3 + faceVertex]
                    = m_originalMesh->getFaceColors(width)[m_faces[face] * 3 + faceVertex];
            }
            if (m_originalMesh->hasFaceNormals())
            {
                faceNormals[face * 3 + faceVertex]
                    = m_originalMesh->getFaceNormals()[m_faces[face] * 3 + faceVertex];
            }*/
        }
    }

    // build new model from newly created buffers
    lvr2::MeshBufferPtr mesh(new lvr2::MeshBuffer);

    mesh->setVertices(vertices, numVertices());
    mesh->setFaceIndices(faceIndices, numFaces());

    // add additional channels
    /*if (m_originalMesh->hasFaceColors())
    {
        mesh->setFaceColors(faceColors, 3);
    }
    if (m_originalMesh->hasVertexColors())
    {
        mesh->setVertexColors(vertexColors, 3);
    }
    if (m_originalMesh->hasFaceNormals())
    {
        mesh->setFaceNormals(faceNormals);
    }
    if (m_originalMesh->hasVertexNormals())
    {
        mesh->setVertexNormals(vertexNormals);
    }*/

    mesh->addAtomic<unsigned int>(m_duplicateVertices.size(), "num_duplicates");

    return mesh;
}

} // namespace lvr2
