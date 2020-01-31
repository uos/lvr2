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
ChunkHashGrid::ChunkHashGrid(std::string hdf5Path, size_t cacheSize, float chunkSize)
    : m_cacheSize(cacheSize)
{
    m_io.open(hdf5Path);

    try
    {
        setChunkSize(m_io.loadChunkSize());
        setBoundingBox(m_io.loadBoundingBox());
    }
    catch (...)
    {
        setChunkSize(chunkSize);
    }
}

ChunkHashGrid::ChunkHashGrid(std::string hdf5Path,
                             size_t cacheSize,
                             BoundingBox<BaseVector<float>> boundingBox,
                             float chunkSize)
    : m_cacheSize(cacheSize)
{
    m_io.open(hdf5Path);
    setChunkSize(chunkSize);
    setBoundingBox(boundingBox);
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

void ChunkHashGrid::rehashCache(const BaseVector<std::size_t>& oldChunkAmount,
                                const BaseVector<std::size_t>& oldChunkIndexOffset)
{
    std::unordered_map<std::string, std::unordered_map<std::size_t, val_type>> tmpCache;
    for (std::pair<std::string, std::size_t>& elem : m_items)
    {
        // undo old hash function
        int k = elem.second % oldChunkAmount.z - oldChunkIndexOffset.z;
        int j = (elem.second / oldChunkAmount.z) % oldChunkAmount.y - oldChunkIndexOffset.y;
        int i = elem.second / (oldChunkAmount.y * oldChunkAmount.z) - oldChunkIndexOffset.x;

        val_type chunk;

        auto layerIt = tmpCache.find(elem.first);
        if (layerIt != tmpCache.end())
        {
            auto chunkIt = tmpCache[elem.first].find(elem.second);
            if (chunkIt != tmpCache[elem.first].end())
            {
                chunk = tmpCache[elem.first][elem.second];
            }
            else
            {
                chunk = m_hashGrid[elem.first][elem.second];
            }
        }
        else
        {
            chunk = m_hashGrid[elem.first][elem.second];
        }

        std::size_t newHash = hashValue(i, j, k);

        auto chunkIt = m_hashGrid[elem.first].find(newHash);
        if (chunkIt != m_hashGrid[elem.first].end())
        {
            tmpCache[elem.first][newHash] = m_hashGrid[elem.first][newHash];
        }
        m_hashGrid[elem.first][newHash] = chunk;
        m_hashGrid[elem.first].erase(elem.second);

        elem.second = newHash;
    }
}

void ChunkHashGrid::expandBoundingBox(const val_type& data)
{
    FloatChannelOptional geometryChannel = boost::apply_visitor(ChunkGeomtryChannelVisitor(), data);
    if (geometryChannel)
    {
        BoundingBox<BaseVector<float>> boundingBox = m_boundingBox;
        for (unsigned int i = 0; i < geometryChannel.get().numElements(); i++)
        {
            boundingBox.expand(static_cast<BaseVector<float>>(geometryChannel.get()[i]));
        }

        setBoundingBox(boundingBox);
    }
}

void ChunkHashGrid::loadChunk(std::string layer, int x, int y, int z, const val_type& data)
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

void ChunkHashGrid::setBoundingBox(const BoundingBox<BaseVector<float>> boundingBox)
{
    if (m_boundingBox.getMin() == boundingBox.getMin()
        && m_boundingBox.getMax() == boundingBox.getMax())
    {
        return;
    }

    m_boundingBox = boundingBox;
    m_io.saveBoundingBox(m_boundingBox);

    BaseVector<std::size_t> chunkIndexOffset;
    chunkIndexOffset.x
        = static_cast<std::size_t>(std::ceil(-getBoundingBox().getMin().x / getChunkSize()) + 1);
    chunkIndexOffset.y
        = static_cast<std::size_t>(std::ceil(-getBoundingBox().getMin().y / getChunkSize()) + 1);
    chunkIndexOffset.z
        = static_cast<std::size_t>(std::ceil(-getBoundingBox().getMin().z / getChunkSize()) + 1);

    BaseVector<std::size_t> chunkAmount;
    chunkAmount.x
        = static_cast<std::size_t>(std::ceil(getBoundingBox().getXSize() / getChunkSize()) + 1);
    chunkAmount.y
        = static_cast<std::size_t>(std::ceil(getBoundingBox().getYSize() / getChunkSize()) + 1);
    chunkAmount.z
        = static_cast<std::size_t>(std::ceil(getBoundingBox().getZSize() / getChunkSize()) + 1);

    setChunkAmountAndOffset(chunkAmount, chunkIndexOffset);
}

void ChunkHashGrid::setChunkAmountAndOffset(const BaseVector<std::size_t>& chunkAmount,
                                            const BaseVector<std::size_t>& chunkIndexOffset)
{
    if (m_chunkAmount != chunkAmount || m_chunkIndexOffset != chunkIndexOffset)
    {
        BaseVector<std::size_t> oldChunkAmount      = m_chunkAmount;
        BaseVector<std::size_t> oldChunkIndexOffset = m_chunkIndexOffset;

        m_chunkAmount      = chunkAmount;
        m_chunkIndexOffset = chunkIndexOffset;

        rehashCache(oldChunkAmount, oldChunkIndexOffset);
    }
}

} /* namespace lvr2 */
