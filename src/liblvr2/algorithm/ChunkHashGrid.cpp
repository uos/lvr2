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
* ChunkHashGrid.cpp
*
* @date 17.09.2019
* @author Raphael Marx
*/


#include "lvr2/algorithm/ChunkHashGrid.hpp"

namespace lvr2
{
    ChunkHashGrid::ChunkHashGrid(std::string hdf5Path)
    :m_chunkIO(std::shared_ptr<ChunkIO>(new ChunkIO(hdf5Path)))
    {
    }

bool ChunkHashGrid::loadChunk(size_t cellIndex)
{
    lvr2::MeshBufferPtr chunk = m_chunkIO->loadChunk(cellIndex);
    if(chunk.get()){
        m_hashGrid.insert({cellIndex, chunk});
        return true;
    }
    return false;
}

MeshBufferPtr ChunkHashGrid::findChunk(size_t cellIndex)
{
    // try to find mesh in hashGrid
    auto it = m_hashGrid.find(cellIndex);
    if (it != m_hashGrid.end())
    {
        return it->second;
    }

    // if the chunk isn't loaded yet, we search for it in the hdf5 file
    if(loadChunk(cellIndex))
    {
        return m_hashGrid.find(cellIndex)->second;
    }

    // if the chunkIndex has no mesh, we return an empty mesh. Is there a better way to do this?
    MeshBufferPtr ret;
    return ret;
}
} /* namespace lvr2 */
