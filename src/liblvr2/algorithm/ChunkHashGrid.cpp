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
ChunkHashGrid::ChunkHashGrid(std::string hdf5Path, size_t cacheSize)
:m_chunkIO(std::shared_ptr<ChunkIO>(new ChunkIO(hdf5Path))),
m_cacheSize(cacheSize)
{
}

bool ChunkHashGrid::loadChunk(size_t hashValue, int x, int y, int z)
{
    std::string chunkName = std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
    lvr2::MeshBufferPtr chunk = m_chunkIO->loadChunk(chunkName);
    if(chunk.get()){
        set(hashValue, chunk);
        return true;
    }
    return false;
}

MeshBufferPtr ChunkHashGrid::findChunk(size_t hashValue, int x, int y, int z)
{
    MeshBufferPtr found;
    // try to load mesh from hash map
    if(get(hashValue, found))
    {
        return found;
    }
    // otherwise try to load the chunk from the hdf5
    if(loadChunk(hashValue, x, y, z))
    {
        get(hashValue, found);
    }
    return found;
}

MeshBufferPtr ChunkHashGrid::findChunkCondition(size_t hashValue, int x, int y, int z, std::string channelName)
{
    MeshBufferPtr found = findChunk(hashValue, x, y, z);
    if(found)
    {
        if(found->hasIndexChannel(channelName) || found->hasFloatChannel(channelName) || found->hasUCharChannel(channelName)) //templating needed?
        {
            return found;
        }
        MeshBufferPtr empty;
        return empty;
    }
    return found;
}

void ChunkHashGrid::set(size_t hashValue, const MeshBufferPtr& mesh)
{
    auto it = m_hashGrid.find(hashValue);
    if(it == m_hashGrid.end())
    {
        items.push_front(hashValue);
        m_hashGrid[hashValue] = {mesh, items.begin()};
        if (m_hashGrid.size() > m_cacheSize)
        {
            m_hashGrid.erase(items.back());
            items.pop_back();
        }
    }
    else
    {
        items.erase((it->second.second));
        items.push_front(hashValue);
        m_hashGrid[hashValue] = {mesh, items.begin()};
    }
}
bool ChunkHashGrid::get(size_t hashValue, MeshBufferPtr& mesh)
{
    auto it = m_hashGrid.find(hashValue);
    if(it != m_hashGrid.end())
    {
        items.erase(it->second.second);
        items.push_front(hashValue);
        m_hashGrid[hashValue] = {it->second.first, items.begin()};
        mesh = it->second.first;
        return true;
    }
    // return false, because the chunk doesn't exist in hashMap
    return false;
}

} /* namespace lvr2 */
