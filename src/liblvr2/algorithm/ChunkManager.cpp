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

namespace lvr2
{

ChunkManager::ChunkManager(lvr2::ModelPtr model)
{
    m_originalModel = model;
    findRange();
}

void ChunkManager::findRange()
{
    // initialize with minimum and maximum values.
    m_maxX = std::numeric_limits<float>::lowest();
    m_maxY = std::numeric_limits<float>::lowest();
    m_maxZ = std::numeric_limits<float>::lowest();
    m_minX = std::numeric_limits<float>::max();
    m_minY = std::numeric_limits<float>::max();
    m_minZ = std::numeric_limits<float>::max();

    //go through the mesh and find the minimum and maximum values
    for(size_t i = 0; i < m_originalModel->m_mesh->numVertices(); i++)
    {
        m_minX = std::min(m_minX, m_originalModel->m_mesh->getVertices()[3*i]);
        m_minY = std::min(m_minY, m_originalModel->m_mesh->getVertices()[3*i+1]);
        m_minZ = std::min(m_minZ, m_originalModel->m_mesh->getVertices()[3*i+2]);
        m_maxX = std::max(m_maxX, m_originalModel->m_mesh->getVertices()[3*i]);
        m_maxY = std::max(m_maxY, m_originalModel->m_mesh->getVertices()[3*i+1]);
        m_maxZ = std::max(m_maxZ, m_originalModel->m_mesh->getVertices()[3*i+2]);
    }
}

void ChunkManager::chunk(float chunksize, std::string savePath)
{
    boost::shared_array<std::shared_ptr<std::vector<unsigned int>>> vertexUse(new std::shared_ptr<std::vector<unsigned int>>[m_originalModel->m_mesh->numVertices()]);

    // compute number of chunks for each dimension
    int amountX = (int) std::ceil((m_maxX - m_minX) / chunksize);
    int amountY = (int) std::ceil((m_maxY - m_minY) / chunksize);
    int amountZ = (int) std::ceil((m_maxZ - m_minZ) / chunksize);

    m_chunkedFaces.resize(amountX * amountY * amountZ, std::vector<unsigned int>());
    std::vector<unsigned int> numVertices(amountX * amountY * amountZ, 0);
  
    // assign the faces to the chunks
    for(int i = 0; i < m_originalModel->m_mesh->numFaces(); i++)
    {
        float x;
        float y;
        float z;

        getCenter(m_originalModel->m_mesh->getFaceIndices()[i * 3], m_originalModel->m_mesh->getFaceIndices()[i * 3 + 1], m_originalModel->m_mesh->getFaceIndices()[i * 3 + 2], x, y, z);
        addFace(i, (int) ((x - m_minX) / chunksize) * amountY * amountZ + (int) ((y - m_minY) / chunksize) * amountZ + (int) ((z - m_minZ) / chunksize), vertexUse, numVertices);
    }

    // save the chunks as .ply  
    lvr2::ModelFactory mf;
    unsigned int chunkID;
    for (int i = 0; i < amountX; i++)
    {
        for (int j = 0; j < amountY; j++)
        {
            for (int k = 0; k < amountZ; k++)
            {
                chunkID = i * amountY * amountZ + j * amountZ + k;
                if (!m_chunkedFaces[chunkID].empty())
                {
                    std::cout << "writing " << i << " " << j << " " << k << std::endl;
                    Chunk chunk(m_originalModel, m_chunkedFaces[chunkID], numVertices[chunkID]);

                    mf.saveModel(chunk.getModel(), savePath + "/" + std::to_string(i) + "-" + std::to_string(j) + "-" + std::to_string(k) + ".ply");
                }
            }
        }
    }
}

void ChunkManager::getCenter(unsigned int index0, unsigned int index1, unsigned int index2, float& x, float& y, float& z)
{
    x = (m_originalModel->m_mesh->getVertices()[index0 * 3 + 0]
            + m_originalModel->m_mesh->getVertices()[index1 * 3 + 0]
            + m_originalModel->m_mesh->getVertices()[index2 * 3 + 0]) / 3;
    y = (m_originalModel->m_mesh->getVertices()[index0 * 3 + 1]
            + m_originalModel->m_mesh->getVertices()[index1 * 3 + 1]
            + m_originalModel->m_mesh->getVertices()[index2 * 3 + 1]) / 3;
    z = (m_originalModel->m_mesh->getVertices()[index0 * 3 + 2]
            + m_originalModel->m_mesh->getVertices()[index1 * 3 + 2]
            + m_originalModel->m_mesh->getVertices()[index2 * 3 + 2]) / 3;
}


void ChunkManager::addFace(unsigned int index, unsigned int chunkID, boost::shared_array<std::shared_ptr<std::vector<unsigned int>>>& vertexUse, std::vector<unsigned int>& numVertices)
{
    // add the original index of the face to the chunk with its chunkID
    m_chunkedFaces[chunkID].push_back(index);

    // next add the vertices of the face to the chunk
    for (uint8_t i = 0; i < 3; i++)
    {
        // if the chunk has not been initialized, we make a vector to store the vertex indices
        if (vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]] == nullptr)
        {
            vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]] = std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);
        }

        // if the vertex is not in the vector, we add the vertex (we just mark the vertex by adding the chunkID)
        if(std::find(vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->begin(), vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->end(), chunkID) == vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->end())
        {
            vertexUse[m_originalModel->m_mesh->getFaceIndices()[index * 3 + i]]->push_back(chunkID);
            numVertices[chunkID]++; 
        }
    }
}

} /* namespace lvr2 */
