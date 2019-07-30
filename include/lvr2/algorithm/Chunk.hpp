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
 * Chunk.hpp
 *
 * @date
 * @author
 */

#ifndef CHUNK_HPP
#define CHUNK_HPP

#include "lvr2/io/Model.hpp"

namespace lvr2
{

class Chunk
{
    public:
        Chunk(lvr2::ModelPtr originalModel, std::vector<unsigned int> faceIndices, unsigned int numVertices);

        ~Chunk();

        /// build mesh of chunk
        lvr2::ModelPtr getModel();

        unsigned int numFaces();
        lvr2::indexArray getFaceIndices();
        lvr2::floatArr getVertices();

        
    private:

        /// model that is being chunked
        lvr2::ModelPtr m_originalModel = nullptr;

        unsigned int m_numVertices;

        lvr2::floatArr m_vertices;
        lvr2::indexArray m_faceIndices;

        /// indices of faces in original model
        std::vector<unsigned int> m_faces;


};

} /* namespace lvr2 */

#endif // CHUNK_HPP
