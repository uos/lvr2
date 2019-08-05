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
 * ChunkManager.cpp
 *
 * @date 21.07.2019
 * @author Malte kl. Piening
 * @author Marcel Wiegand
 */

#include <cmath>

#include "lvr2/algorithm/ChunkManager.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/algorithm/ChunkBuilder.hpp"

namespace lvr2
{

ChunkManager::ChunkManager(MeshBufferPtr mesh, float chunksize, std::string savePath) : m_chunkSize(chunksize)
{
    initBoundingBox(mesh);

    // compute number of chunks for each dimension
    m_amount.x = (std::size_t) std::ceil(m_boundingBox.getXSize() / m_chunkSize);
    m_amount.y = (std::size_t) std::ceil(m_boundingBox.getYSize() / m_chunkSize);
    m_amount.z = (std::size_t) std::ceil(m_boundingBox.getZSize() / m_chunkSize);

    buildChunks(mesh, savePath);
}

MeshBufferPtr ChunkManager::extractArea(const BoundingBox<BaseVector<float>>& area)
{
    std::vector<MeshBufferPtr> chunks;

    // find all required chunks
    // TODO: check if we need + 1
    BaseVector<float> maxSteps = (area.getMax() - area.getMin()) / m_chunkSize;
    for (std::size_t i = 0; i < maxSteps.x; ++i)
    {
        for (std::size_t j = 0; j < maxSteps.y; ++j)
        {
            for (std::size_t k = 0; k < maxSteps.z; ++k)
            {
                std::size_t cellIndex = getCellIndex(
                        area.getMin() + BaseVector<float>(i * m_chunkSize, j * m_chunkSize, k * m_chunkSize));
                // TODO: remove saving tmp chunks later
                ModelFactory::saveModel(lvr2::ModelPtr(new lvr2::Model(m_hashGrid[cellIndex])), "area/" + std::to_string(cellIndex) + ".ply");
                chunks.push_back(m_hashGrid[cellIndex]);
            }
        }
    }

    // TODO: concat chunks
    MeshBufferPtr areaMeshPtr = nullptr;

    return areaMeshPtr;
}

void ChunkManager::initBoundingBox(MeshBufferPtr mesh)
{
    BaseVector<float> currVertex;

    for (std::size_t i = 0; i < mesh->numVertices(); i++)
    {
        currVertex.x = mesh->getVertices()[3 * i];
        currVertex.y = mesh->getVertices()[3 * i + 1];
        currVertex.z = mesh->getVertices()[3 * i + 2];

        m_boundingBox.expand(currVertex);
    }
}

void ChunkManager::buildChunks(MeshBufferPtr mesh, std::string savePath)
{
    // one vector of variable size for each vertex - this is used for duplicate detection
    std::shared_ptr<std::vector<std::vector<std::shared_ptr<ChunkBuilder>>>> vertexUse(new std::vector<std::vector<std::shared_ptr<ChunkBuilder>>>(mesh->numVertices(), std::vector<std::shared_ptr<ChunkBuilder>>()));

    std::vector<std::shared_ptr<ChunkBuilder>> chunkBuilders(m_amount.x * m_amount.y * m_amount.z);

    for (std::size_t i = 0; i < m_amount.x; i++)
    {
        for (std::size_t j = 0; j < m_amount.y; j++)
        {
            for (std::size_t k = 0; k < m_amount.z; k++)
            {
                chunkBuilders[hashValue(i, j, k)] = std::shared_ptr<ChunkBuilder>(new ChunkBuilder(mesh, vertexUse));
            }
        }
    }

    // assign the faces to the chunks
    FloatChannel verticesChannel = *mesh->getFloatChannel("vertices");
    IndexChannel facesChannel = *mesh->getIndexChannel("face_indices");
    BaseVector<float> currentCenterPoint;
    for (std::size_t i = 0; i < mesh->numFaces(); i++)
    {
        currentCenterPoint = getFaceCenter(verticesChannel, facesChannel, i);
        chunkBuilders[getCellIndex(currentCenterPoint)]->addFace(i);
    }

    // save the chunks as .ply
    for (std::size_t i = 0; i < m_amount.x; i++)
    {
        for (std::size_t j = 0; j < m_amount.y; j++)
        {
            for (std::size_t k = 0; k < m_amount.z; k++)
            {
                std::size_t hash = hashValue(i, j, k);

                if (chunkBuilders[hash]->numFaces() > 0)
                {
                    std::cout << "writing " << i << " " << j << " " << k << std::endl;

                    // get mesh of chunk from chunk builder
                    MeshBufferPtr chunkMeshPtr = chunkBuilders[hash]->buildMesh();

                    // insert chunked mesh into hash grid
                    m_hashGrid.insert({hash, chunkMeshPtr});

                    // export chunked meshes for debugging
                    ModelFactory::saveModel(
                        ModelPtr(new Model(chunkMeshPtr)),
                        savePath + "/" + std::to_string(i) + "-" + std::to_string(j) + "-" + std::to_string(k) + ".ply");
                }
            }
        }
    }
}

BaseVector<float> ChunkManager::getFaceCenter(FloatChannel verticesChannel, IndexChannel facesChannel, unsigned int faceIndex)
{
    BaseVector<float> vertex1(verticesChannel[facesChannel[faceIndex][0]]);
    BaseVector<float> vertex2(verticesChannel[facesChannel[faceIndex][1]]);
    BaseVector<float> vertex3(verticesChannel[facesChannel[faceIndex][2]]);

    return (vertex1 + vertex2 + vertex3) / 3;
}

std::size_t ChunkManager::getCellIndex(const BaseVector<float>& vec)
{
    BaseVector<float> tmpVec = (vec - m_boundingBox.getMin()) / m_chunkSize;
    return (std::size_t) tmpVec.x * m_amount.y * m_amount.z + (std::size_t) tmpVec.y * m_amount.z + (std::size_t) tmpVec.z;
}

} /* namespace lvr2 */
