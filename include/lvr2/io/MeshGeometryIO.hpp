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


#ifndef MESHGEOMETRYIO
#define MESHGEOMETRYIO

#include "lvr2/types/BaseBuffer.hpp"

namespace lvr2
{ 

class MeshGeometryIO
{
public:
    /**
     * @brief Persistence layer interface, Accesses the vertices of the mesh in the persistence layer.
     * @return An optional float channel, the channel is valid if the mesh vertices have been read successfully
     */
    virtual FloatChannelOptional getVertices() = 0;

    /**
     * @brief Persistence layer interface, Accesses the face indices of the mesh in the persistence layer.
     * @return An optional index channel, the channel is valid if the mesh indices have been read successfully
     */
    virtual IndexChannelOptional getIndices() = 0;

    /**
     * @brief Persistence layer interface, Writes the vertices of the mesh to the persistence layer.
     * @return true if the channel has been written successfully
     */
    virtual bool addVertices(const FloatChannel& channel_ptr) = 0;

    /**
     * @brief Persistence layer interface, Writes the face indices of the mesh to the persistence layer.
     * @return true if the channel has been written successfully
     */
    virtual bool addIndices(const IndexChannel& channel_ptr) = 0;
};

} // namespace lvr2

#endif