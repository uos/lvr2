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

ChunkBuilder::ChunkBuilder(unsigned int id, lvr2::ModelPtr originalModel, boost::shared_array<std::shared_ptr<std::vector<unsigned int>>> vertexUse) :
    m_id(id),
    m_originalModel(originalModel),
    m_vertexUse(vertexUse)
{

}

ChunkBuilder::~ChunkBuilder()
{

}

void ChunkBuilder::addFace(unsigned int index)
{
    m_faces.push_back(index);

    for (uint8_t i = 0; i < 3; i++)
    {
        if (m_vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]] == nullptr)
        {
            m_vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]] = std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);
        }

        if(std::find(m_vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->begin(), m_vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->end(), m_id) == m_vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->end())
        {
            m_vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->push_back(m_id);
            m_numVertices++;
        }
    }
}

unsigned int ChunkBuilder::numFaces()
{
    return m_faces.size();
}

lvr2::ModelPtr ChunkBuilder::buildMesh()
{
    std::unordered_map<unsigned int, unsigned int> vertexIndices;

    lvr2::floatArr vertices(new float[m_numVertices * 3]);
    lvr2::indexArray faceIndices(new unsigned int[m_faces.size() * 3]);

    unsigned int faceIndexCnt = 0;
    for (unsigned int faceIndex : m_faces)
    {
        for (uint8_t i = 0; i < 3; i++)
        {
            if (vertexIndices.find(m_originalModel->m_mesh->getFaceIndices()[faceIndex * 3 + i]) == vertexIndices.end())
            {
                vertexIndices[m_originalModel->m_mesh->getFaceIndices()[faceIndex * 3 + i]] = vertexIndices.size();

                for (uint8_t j = 0; j < 3; j++)
                {
                    vertices[vertexIndices[m_originalModel->m_mesh->getFaceIndices()[faceIndex * 3 + i]] * 3 + j]
                            = m_originalModel->m_mesh->getVertices()[m_originalModel->m_mesh->getFaceIndices()[faceIndex * 3 + i] * 3 + j];
                }
            }

            faceIndices[faceIndexCnt * 3 + i] = vertexIndices[m_originalModel->m_mesh->getFaceIndices()[faceIndex * 3 + i]];
        }
        faceIndexCnt++;
    }

    lvr2::ModelPtr model(new lvr2::Model(lvr2::MeshBufferPtr(new lvr2::MeshBuffer)));

    model->m_mesh->setVertices(vertices, vertexIndices.size());
    model->m_mesh->setFaceIndices(faceIndices, m_faces.size());

    return model;
}

} /* namespacd lvr2 */
