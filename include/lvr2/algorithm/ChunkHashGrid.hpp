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
 * ChunkHashGrid.hpp
 *
 * @date 17.09.2019
 * @author Raphael Marx
 */

#ifndef CHUNK_HASH_GRID_HPP
#define CHUNK_HASH_GRID_HPP

#include "lvr2/io/GHDF5IO.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/hdf5/ChunkIO.hpp"

#include <list>
#include <unordered_map>

namespace lvr2
{
class ChunkHashGrid
{
  public:
    using val_type = boost::variant<MeshBufferPtr, PointBufferPtr>;

    using io = Hdf5IO<hdf5features::ChunkIO,
                      hdf5features::ArrayIO,
                      hdf5features::ChannelIO,
                      hdf5features::VariantChannelIO,
                      hdf5features::MeshIO>;

    /**
     * @brief class to load chunks from an HDF5 file
     *
     * @param hdf5Path path to the HDF5 file
     */
    explicit ChunkHashGrid(std::string hdf5Path, size_t cacheSize);

    template <typename T>
    void setChunk(std::string layer, int x, int y, int z, T data);

    template <typename T>
    boost::optional<T> getChunk(std::string layer, int x, int y, int z);

    bool isChunkLoaded(std::string layer, size_t hashValue);
    bool isChunkLoaded(std::string layer, int x, int y, int z);

    /**
     * @brief Calculates the hash value for the given index triple
     *
     * @param i index of x-axis
     * @param j index of y-axis
     * @param k index of z-axis
     *
     * @return hash value
     */
    inline std::size_t hashValue(int i, int j, int k) const
    {
        return i * m_chunkAmount.y * m_chunkAmount.z + j * m_chunkAmount.z + k;
    }

    const BoundingBox<BaseVector<float>>& getBoundingBox() const
    {
        return m_boundingBox;
    }

    float getChunkSize() const
    {
        return m_chunkSize;
    }

    const BaseVector<std::size_t>& getChunkAmount() const
    {
        return m_chunkAmount;
    }

  protected:
    template <typename T>
    bool loadChunk(std::string layer, int x, int y, int z);

    bool loadChunk(std::string layer, int x, int y, int z, const val_type& data);

    void setBoundingBox(const BoundingBox<BaseVector<float>> boundingBox)
    {
        m_boundingBox = boundingBox;
        m_io.saveBoundingBox(m_boundingBox);
    }

    void setChunkSize(float chunkSize)
    {
        m_chunkSize = chunkSize;
        m_io.saveChunkSize(m_chunkSize);
    }

    void setChunkAmount(const BaseVector<std::size_t>& chunkAmount)
    {
        m_chunkAmount = chunkAmount;
        m_io.saveAmount(m_chunkAmount);
    }

  private:
    // chunkIO for the HDF5 file-IO
    io m_io;

    // number of chunks that will be cached before deleting old chunks
    size_t m_cacheSize;

    // ordered list to save recently used hashValues for the lru cache
    std::list<std::pair<std::string, size_t>> m_items;

    // hash map containing chunked meshes
    std::unordered_map<std::string, std::unordered_map<size_t, val_type>> m_hashGrid;

    // bounding box of the entire chunked model
    BoundingBox<BaseVector<float>> m_boundingBox;

    // size of chunks
    float m_chunkSize;

    // amount of chunks
    BaseVector<std::size_t> m_chunkAmount;
};

} /* namespace lvr2 */

#include "ChunkHashGrid.tcc"

#endif // CHUNK_HASH_GRID_HPP
