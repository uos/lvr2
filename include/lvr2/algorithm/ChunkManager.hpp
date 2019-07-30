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
 * ChunkManager.hpp
 *
 * @date 
 * @author 
 */

#ifndef CHUNK_MANAGER_HPP
#define CHUNK_MANAGER_HPP

#include "Chunk.hpp"
#include "lvr2/io/Model.hpp"

namespace lvr2 {

class ChunkManager
{
    public:
        ChunkManager(lvr2::ModelPtr model);

        // add each face to one chunk and save all chunks sequentially
        void chunk(float chunksize, std::string savePath);

    private:
        // sets the bounding box for all dimensions in the mesh (minX, maxX, minY, ...)
        void findRange();

        // computes the center of a face with given vertex-indices and saves it in x, y and z
        void getCenter(unsigned int index0, unsigned int index1, unsigned int index2, float& x, float& y, float& z);

        // add the face to the chunk
        void addFace(unsigned int index, unsigned int chunkID, boost::shared_array<std::shared_ptr<std::vector<unsigned int>>>& vertexUse, std::vector<unsigned int>& numVertices);
        
        lvr2::ModelPtr m_originalModel;
        float m_minX, m_minY, m_minZ, m_maxX, m_maxY, m_maxZ;

        // a vector that holds one vector for each chunk, where its faceIndices are stored
        std::vector<std::vector<unsigned int>> m_chunkedFaces;
};

} /* namespace lvr2 */

#endif // CHUNK_MANAGER_HPP
