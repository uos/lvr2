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

ChunkManager::ChunkManager(std::string modelPath)
{
    lvr2::ModelFactory mf;
    m_originalModel = mf.readModel(modelPath);
    findRange();
}

void ChunkManager::findRange()
{
    m_maxX = std::numeric_limits<float>::lowest();
    m_maxY = std::numeric_limits<float>::lowest();
    m_maxZ = std::numeric_limits<float>::lowest();
    m_minX = std::numeric_limits<float>::max();
    m_minY = std::numeric_limits<float>::max();
    m_minZ = std::numeric_limits<float>::max();

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

    boost::shared_array<std::shared_ptr<ChunkBuilder>> m_chunks(new std::shared_ptr<ChunkBuilder>[amountX * amountY * amountZ]);

    for (int i = 0; i < amountX; i++)
    {
        for (int j = 0; j < amountY; j++)
        {
            for (int k = 0; k < amountZ; k++)
            {
                m_chunks[i * amountY * amountZ + j * amountZ + k] = std::shared_ptr<ChunkBuilder>(new ChunkBuilder(i * amountY * amountZ + j * amountZ + k, m_originalModel, vertexUse));
            }
        }
    }

    // assign the faces to the chunks
    for(int i = 0; i < m_originalModel->m_mesh->numFaces(); i++)
    {
        float x;
        float y;
        float z;

        getCenter(m_originalModel->m_mesh->getFaceIndices()[i * 3], m_originalModel->m_mesh->getFaceIndices()[i * 3 + 1], m_originalModel->m_mesh->getFaceIndices()[i * 3 + 2], x, y, z);

        m_chunks[(int) ((x - m_minX) / chunksize) * amountY * amountZ + (int) ((y - m_minY) / chunksize) * amountZ + (int) ((z - m_minZ) / chunksize)]->addFace(i);
    }

    // save the chunks as .ply  
    lvr2::ModelFactory mf;
    for (int i = 0; i < amountX; i++)
    {
        for (int j = 0; j < amountY; j++)
        {
            for (int k = 0; k < amountZ; k++)
            {
                if (m_chunks[i * amountY * amountZ + j * amountZ + k]->numFaces() > 0)
                {
                    std::cout << "writing " << i << " " << j << " " << k << std::endl;
                    mf.saveModel(m_chunks[i * amountY * amountZ + j * amountZ + k]->buildMesh(), savePath + "/" + std::to_string(i) + "-" + std::to_string(j) + "-" + std::to_string(k) + ".ply");
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

} /* namespace lvr2 */
