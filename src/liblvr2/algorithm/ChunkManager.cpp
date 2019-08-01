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

ChunkManager::ChunkManager(MeshBufferPtr mesh, float chunksize, std::string savePath)
{
    initBoundingBox(mesh);
    buildChunks(mesh, chunksize, savePath);
}

void ChunkManager::initBoundingBox(MeshBufferPtr mesh)
{
    BaseVector<float> currVertex;

    for(size_t i = 0; i < mesh->numVertices(); i++)
    {
        currVertex.x = mesh->getVertices()[3 * i];
        currVertex.y = mesh->getVertices()[3 * i + 1];
        currVertex.z = mesh->getVertices()[3 * i + 2];

        m_boundingBox.expand(currVertex);
    }
}

void ChunkManager::buildChunks(MeshBufferPtr mesh, float chunksize, std::string savePath)
{
    // compute number of chunks for each dimension
    int amountX = (int) std::ceil((m_boundingBox.getMax().x - m_boundingBox.getMin().x) / chunksize);
    int amountY = (int) std::ceil((m_boundingBox.getMax().y - m_boundingBox.getMin().y) / chunksize);
    int amountZ = (int) std::ceil((m_boundingBox.getMax().z - m_boundingBox.getMin().z) / chunksize);

    // one vector of variable size for each vertex - this is used for duplicate detection
    std::shared_ptr<std::vector<std::vector<std::shared_ptr<ChunkBuilder>>>> vertexUse(new std::vector<std::vector<std::shared_ptr<ChunkBuilder>>>(mesh->numVertices(), std::vector<std::shared_ptr<ChunkBuilder>>()));

    std::vector<std::shared_ptr<ChunkBuilder>> chunkBuilders(amountX * amountY * amountZ);

    for (int i = 0; i < amountX; i++)
    {
        for (int j = 0; j < amountY; j++)
        {
            for (int k = 0; k < amountZ; k++)
            {
                chunkBuilders[i * amountY * amountZ + j * amountZ + k] = std::shared_ptr<ChunkBuilder>(new ChunkBuilder(mesh, vertexUse));
            }
        }
    }

    // assign the faces to the chunks
    FloatChannel verticesChannel = *mesh->getFloatChannel("vertices");
    IndexChannel facesChannel = *mesh->getIndexChannel("face_indices");
    BaseVector<float> currentCenterPoint;
    for(int i = 0; i < mesh->numFaces(); i++)
    {
        currentCenterPoint = getFaceCenter(verticesChannel, facesChannel, i);

        chunkBuilders[(int) ((currentCenterPoint.x - m_boundingBox.getMin().x) / chunksize) * amountY * amountZ
                + (int) ((currentCenterPoint.y - m_boundingBox.getMin().y) / chunksize) * amountZ
                + (int) ((currentCenterPoint.z - m_boundingBox.getMin().z) / chunksize)]->addFace(i);
    }

    // save the chunks as .ply
    for (int i = 0; i < amountX; i++)
    {
        for (int j = 0; j < amountY; j++)
        {
            for (int k = 0; k < amountZ; k++)
            {
                std::size_t hash = i * amountY * amountZ + j * amountZ + k;

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

} /* namespace lvr2 */
