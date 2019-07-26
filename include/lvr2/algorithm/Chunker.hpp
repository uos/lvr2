#ifndef CHUNKER_HPP
#define CHUNKER_HPP

#include "ChunkBuilder.hpp"
#include "lvr2/io/Model.hpp"

namespace lvr2 {

class Chunker
{
    public:
        Chunker(lvr2::ModelPtr model);
        Chunker(std::string modelPath);

        // add each face to one chunk and save all chunks sequentially
        void chunk(float chunksize, std::string savePath);

    private:
        void findRange();

        // computes the center of a face
        void getCenter(unsigned int index0, unsigned int index1, unsigned int index2, float& x, float& y, float& z);
        
        lvr2::ModelPtr m_originalModel;
        float m_minX, m_minY, m_minZ, m_maxX, m_maxY, m_maxZ;
};

} /* namespace lvr2 */

#endif // CHUNKER_HPP
