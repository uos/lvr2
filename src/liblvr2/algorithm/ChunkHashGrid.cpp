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
ChunkHashGrid::ChunkHashGrid(std::string hdf5Path, size_t cacheSize) : m_cacheSize(cacheSize)
{
    m_io.open(hdf5Path);

    try
    {
        m_boundingBox = m_io.loadBoundingBox();
        m_chunkSize   = m_io.loadChunkSize();
        m_chunkAmount = m_io.loadAmount();
    }
    catch (std::runtime_error& e)
    {
    }
}

bool ChunkHashGrid::isChunkLoaded(std::string layer, std::size_t hashValue)
{
    auto layerIt = m_hashGrid.find(layer);
    if (layerIt != m_hashGrid.end())
    {
        auto chunkIt = layerIt->second.find(hashValue);
        if (chunkIt != layerIt->second.end())
        {
            return true;
        }
    }
    return false;
}

bool ChunkHashGrid::isChunkLoaded(std::string layer, int x, int y, int z)
{
    return isChunkLoaded(layer, hashValue(x, y, z));
}

bool ChunkHashGrid::loadChunk(std::string layer, int x, int y, int z, const val_type& data)
{
    std::size_t chunkHash = hashValue(x, y, z);

    if (isChunkLoaded(layer, chunkHash))
    {
        // chunk exists for layer in grid
        // remove chunk from cache to put the chunk to front of lru cache later
        m_items.remove({layer, chunkHash});
    }

    // add new chunk to cache
    m_items.push_front({layer, chunkHash});

    // check if cache is full
    if (m_items.size() > m_cacheSize)
    {
        // remove chunk from grid keep the grid for the current layer even if it holds no elements
        m_hashGrid[m_items.back().first].erase(m_items.back().second);

        // remove erased element from cache
        m_items.pop_back();
    }

    m_hashGrid[layer][chunkHash] = data;
}

} /* namespace lvr2 */
