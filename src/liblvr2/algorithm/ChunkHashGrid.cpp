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
    : m_chunkIO(std::shared_ptr<ChunkIO>(new ChunkIO(hdf5Path))), m_cacheSize(cacheSize)
{
}

bool ChunkHashGrid::loadChunk(std::string layer, size_t hashValue, int x, int y, int z)
{
    std::string chunkName = std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
    lvr2::MeshBufferPtr chunk = m_chunkIO->loadChunk(chunkName);
    if (chunk.get())
    {
        set(layer, hashValue, chunk);
        return true;
    }
    return false;
}

bool ChunkHashGrid::loadMeshChunk(size_t hashValue, int x, int y, int z)
{
    std::string chunkName = std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
    lvr2::MeshBufferPtr chunk = m_chunkIO->loadChunk(chunkName);
    if (chunk.get())
    {
        set("mesh", hashValue, chunk);
        return true;
    }
    return false;
}

ChunkHashGrid::val_type
ChunkHashGrid::findVariantChunk(std::string layer, size_t hashValue, int x, int y, int z)
{
    val_type found;
    // try to load mesh from hash map
    if (get(layer, hashValue, found))
    {
        return found;
    }
    // otherwise try to load the chunk from the hdf5
    if (loadChunk(layer, hashValue, x, y, z))
    {
        get(layer, hashValue, found);
    }
    return found;
}


MeshBufferPtr ChunkHashGrid::findMeshChunk(size_t hashValue, int x, int y, int z)
{
    return findChunk<MeshBufferPtr>("mesh", hashValue, x, y, z);
}

MeshBufferPtr ChunkHashGrid::findMeshChunkCondition(size_t hashValue, int x, int y, int z, std::string channelName)
{
    MeshBufferPtr found = findMeshChunk(hashValue, x, y, z);
    if (found)
    {
        if (found->hasIndexChannel(channelName) || found->hasFloatChannel(channelName)
            || found->hasUCharChannel(channelName)) // templating needed?
        {
            return found;
        }
        return nullptr;
    }
    return found;
}

void ChunkHashGrid::set(std::string layer, size_t hashValue, const val_type& mesh)
{
    auto layerIt = m_hashGrid.find(layer);
    if (layerIt != m_hashGrid.end())
    {
        auto chunkIt = layerIt->second.find(hashValue);
        if (chunkIt != layerIt->second.end())
        {
            // chunk exists for layer in grid
            // remove chunk from cache to put the chunk to front of lru cache later
            items.remove({layer, hashValue});
        }
    }

    // add new chunk to cache
    items.push_front({layer, hashValue});

    // check if cache is full
    if (items.size() > m_cacheSize)
    {
        // remove chunk from grid keep the grid for the current layer even if it holds no elements
        m_hashGrid[items.back().first].erase(items.back().second);

        // remove erased element from cache
        items.pop_back();
    }

    m_hashGrid[layer][hashValue] = mesh;
}

bool ChunkHashGrid::get(std::string layer, size_t hashValue, val_type& mesh)
{
    auto layerIt = m_hashGrid.find(layer);
    if (layerIt != m_hashGrid.end())
    {
        auto chunkIt = layerIt->second.find(hashValue);
        if (chunkIt != layerIt->second.end())
        {
            // move chunk to the front of the cache queue
            items.remove({layer, hashValue});
            items.push_front({layer, hashValue});

            mesh = chunkIt->second;
            return true;
        }
    }
    // return false, because the chunk doesn't exist in hashMap
    return false;
}

} /* namespace lvr2 */
