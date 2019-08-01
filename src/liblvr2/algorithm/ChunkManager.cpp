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
 * @date
 * @author
 */

#include <cmath>

#include "lvr2/algorithm/ChunkManager.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/algorithm/ChunkBuilder.hpp"

namespace lvr2
{

ChunkManager::ChunkManager(ModelPtr model, float chunksize, std::string savePath)
{
    initBoundingBox(model);
    buildChunks(model, chunksize, savePath);
}

void ChunkManager::initBoundingBox(ModelPtr model)
{
    BaseVector<float> currVertex;

    for(size_t i = 0; i < model->m_mesh->numVertices(); i++)
    {
        currVertex.x = model->m_mesh->getVertices()[3 * i];
        currVertex.y = model->m_mesh->getVertices()[3 * i + 1];
        currVertex.z = model->m_mesh->getVertices()[3 * i + 2];

        m_boundingBox.expand(currVertex);
    }
}

void ChunkManager::buildChunks(ModelPtr model, float chunksize, std::string savePath)
{
    // compute number of chunks for each dimension
    int amountX = (int) std::ceil((m_boundingBox.getMax().x - m_boundingBox.getMin().x) / chunksize);
    int amountY = (int) std::ceil((m_boundingBox.getMax().y - m_boundingBox.getMin().y) / chunksize);
    int amountZ = (int) std::ceil((m_boundingBox.getMax().z - m_boundingBox.getMin().z) / chunksize);

    // one vector of variable size for each vertex - this is used for duplicate detection
    std::shared_ptr<std::vector<std::vector<unsigned int>>> vertexUse(new std::vector<std::vector<unsigned int>>(model->m_mesh->numVertices(), std::vector<unsigned int>()));

    std::vector<std::shared_ptr<ChunkBuilder>> chunkBuilders(amountX * amountY * amountZ);

    for (int i = 0; i < amountX; i++)
    {
        for (int j = 0; j < amountY; j++)
        {
            for (int k = 0; k < amountZ; k++)
            {
                chunkBuilders[i * amountY * amountZ + j * amountZ + k] = std::shared_ptr<ChunkBuilder>(new ChunkBuilder(i * amountY * amountZ + j * amountZ + k, model, vertexUse));
            }
        }
    }

    // assign the faces to the chunks
    BaseVector<float> currentCenterPoint;
    for(int i = 0; i < model->m_mesh->numFaces(); i++)
    {
        currentCenterPoint = getCenter(model, model->m_mesh->getFaceIndices()[i * 3],
                model->m_mesh->getFaceIndices()[i * 3 + 1],
                model->m_mesh->getFaceIndices()[i * 3 + 2]);

        chunkBuilders[(int) ((currentCenterPoint.x - m_boundingBox.getMin().x) / chunksize) * amountY * amountZ
                + (int) ((currentCenterPoint.y - m_boundingBox.getMin().y) / chunksize) * amountZ
                + (int) ((currentCenterPoint.z - m_boundingBox.getMin().z) / chunksize)]->addFace(i);
    }

    // save the chunks as .ply
    ModelFactory mf;
    for (int i = 0; i < amountX; i++)
    {
        for (int j = 0; j < amountY; j++)
        {
            for (int k = 0; k < amountZ; k++)
            {
                if (chunkBuilders[i * amountY * amountZ + j * amountZ + k]->numFaces() > 0)
                {
                    // TODO: dont simply save the chunks as ply files and insert them into the hashgrid

                    std::cout << "writing " << i << " " << j << " " << k << std::endl;
                    mf.saveModel(ModelPtr(new Model(chunkBuilders[i * amountY * amountZ + j * amountZ + k]->buildMesh())), savePath + "/" + std::to_string(i) + "-" + std::to_string(j) + "-" + std::to_string(k) + ".ply");
                }
            }
        }
    }
}

BaseVector<float> ChunkManager::getCenter(ModelPtr model, unsigned int index0, unsigned int index1, unsigned int index2)
{
    float x = (model->m_mesh->getVertices()[index0 * 3 + 0]
            + model->m_mesh->getVertices()[index1 * 3 + 0]
            + model->m_mesh->getVertices()[index2 * 3 + 0]) / 3;

    float y = (model->m_mesh->getVertices()[index0 * 3 + 1]
            + model->m_mesh->getVertices()[index1 * 3 + 1]
            + model->m_mesh->getVertices()[index2 * 3 + 1]) / 3;

    float z = (model->m_mesh->getVertices()[index0 * 3 + 2]
          + model->m_mesh->getVertices()[index1 * 3 + 2]
          + model->m_mesh->getVertices()[index2 * 3 + 2]) / 3;

    return BaseVector<float>(x, y, z);
}

} /* namespace lvr2 */
