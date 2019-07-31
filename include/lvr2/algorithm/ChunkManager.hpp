#ifndef CHUNK_MANAGER_HPP
#define CHUNK_MANAGER_HPP

#include "lvr2/io/Model.hpp"

namespace lvr2 {

class ChunkManager
{
    public:
        ChunkManager(lvr2::ModelPtr model, float chunksize, std::string savePath);

    private:
        // sets the bounding box for all dimensions in the mesh (minX, maxX, minY, ...)
        void findRange();

        // add each face to one chunk and save all chunks sequentially
        void chunk(float chunksize, std::string savePath);

        // computes the center of a face with given vertex-indices and saves it in x, y and z
        void getCenter(unsigned int index0, unsigned int index1, unsigned int index2, float& x, float& y, float& z);
        
        lvr2::ModelPtr m_originalModel;
        float m_minX, m_minY, m_minZ, m_maxX, m_maxY, m_maxZ;
};

} /* namespace lvr2 */

#endif // CHUNKER_HPP
