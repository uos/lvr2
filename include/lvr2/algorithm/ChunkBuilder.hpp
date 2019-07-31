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
 * ChunkBuilder.hpp
 *
 * @date 21.07.2019
 * @author Malte kl. Piening
 */

#ifndef CHUNK_BUILDER_HPP
#define CHUNK_BUILDER_HPP

#include "lvr2/io/Model.hpp"

#include <unordered_map>

namespace lvr2
{

class ChunkBuilder
{
    public:
        ChunkBuilder(unsigned int id, lvr2::ModelPtr originalModel, boost::shared_array<std::shared_ptr<std::vector<unsigned int>>> vertexUse);

        ~ChunkBuilder();

        /// add a face to this chunk
        void addFace(unsigned int index);

        /// build mesh of chunk
        lvr2::ModelPtr buildMesh();

        unsigned int numFaces();
        
    private:
        unsigned int m_id;

        /// model that is being chunked
        lvr2::ModelPtr m_originalModel = nullptr;

        unsigned int m_numVertices = 0;

        /// indices of faces in original model
        std::vector<unsigned int> m_faces;

        boost::shared_array<std::shared_ptr<std::vector<unsigned int>>> m_vertexUse;
};

} /* namespace lvr2 */

#endif // CHUNK_BUILDER_HPP
