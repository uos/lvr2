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

#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/hdf5/ChunkIO.hpp"

#include <list>
#include <unordered_map>

namespace lvr2
{

/**
 * @brief visitor that returns the channel that holds geometic information (like vertices-channel
 * for meshes)
 */
class ChunkGeomtryChannelVisitor : public boost::static_visitor<FloatChannelOptional>
{
  public:
    FloatChannelOptional operator()(const MeshBufferPtr mesh) const
    {
        return mesh->getFloatChannel("vertices");
    }

    FloatChannelOptional operator()(const PointBufferPtr points) const
    {
        return points->getFloatChannel("points");
    }
};

class ChunkHashGrid
{
  public:
    using val_type = boost::variant<MeshBufferPtr, PointBufferPtr>;

    using io = Hdf5Build<hdf5features::ChunkIO>;

    /**
     * @brief class to load chunks from an HDF5 file
     *
     * @param hdf5Path path to the HDF5 file
     */
    explicit ChunkHashGrid(std::string hdf5Path, size_t cacheSize, float chunkSize = 10.0f);

    /**
     * @brief class to load chunks from an HDF5 file
     *
     * @param hdf5Path path to the HDF5 file
     */
    ChunkHashGrid(std::string hdf5Path,
                  size_t cacheSize,
                  BoundingBox<BaseVector<float>> boundingBox,
                  float chunkSize);

    /**
     * @brief sets a chunk of a given layer in hashgrid
     *
     * Adds a chunk for a given layer and coordinate in chunk coordinates to local cache and stores
     * it permanently using io module. Uses the geometrical data of the chunk data (if exists) to
     * expand boundingbox
     *
     * @tparam T type of chunk
     * @param layer layer of chunk
     * @param x x coordinate of chunk in chunk coordinates
     * @param y y coordinate of chunk in chunk coordinates
     * @param z z coordinate of chunk in chunk coordinates
     * @param data content of chunk to be added
     */
    template <typename T>
    void setGeometryChunk(std::string layer, int x, int y, int z, T data);

    /**
     * @brief sets a chunk of a given layer in hashgrid
     *
     * Adds a chunk for a given layer and coordinate in chunk coordinates to local cache and stores
     * it permanently using io module. if the given chunk coordinate does not exist, the boudnig box
     * will be expanded to fit the new chunk
     *
     * @tparam T type of chunk
     * @param layer layer of chunk
     * @param x x coordinate of chunk in chunk coordinates
     * @param y y coordinate of chunk in chunk coordinates
     * @param z z coordinate of chunk in chunk coordinates
     * @param data content of chunk to be added
     */
    template <typename T>
    void setChunk(std::string layer, int x, int y, int z, T data);

    /**
     * @brief delivers the content of a chunk
     *
     * Returns a the content of a chunk from the local cache.
     * If the requested chunk is not cached, the chunk will be loaded from the persistent storage
     * and returned after being added to the cache.
     *
     * @tparam T type of requested chunk
     * @param layer layer of requested chunk
     * @param x x coordinate of chunk in chunk coordinates
     * @param y y coordinate of chunk in chunk coordinates
     * @param z z coordinate of chunk in chunk coordinates
     *
     * @return content of the chunk
     */
    template <typename T>
    boost::optional<T> getChunk(std::string layer, int x, int y, int z);

    /**
     * @brief indicates if wether or not a chunk is currently loaded in the local cache
     *
     * @param layer layer of chunk
     * @param hashValue hashValue hash of the chunk coordinate
     *
     * @return true if chunk is loaded; else false
     */
    bool isChunkLoaded(std::string layer, size_t hashValue);

    /**
     * @brief indicates if wether or not a chunk is currently loaded in the local cache
     *
     * @param layer layer of chunk
     * @param x x coordinate of chunk in chunk coordinates
     * @param y y coordinate of chunk in chunk coordinates
     * @param z z coordinate of chunk in chunk coordinates
     *
     * @return true if chunk is loaded; else false
     */
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
        return (i + m_chunkIndexOffset.x) * m_chunkAmount.y * m_chunkAmount.z
               + (j + m_chunkIndexOffset.y) * m_chunkAmount.z + k + m_chunkIndexOffset.z;
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

    /**
     * @brief returns the minimum chunk ids
     */
    BaseVector<int> getChunkMinChunkIndex() const
    {
        BaseVector<int> offset(m_chunkIndexOffset.x, m_chunkIndexOffset.y, m_chunkIndexOffset.z);
        return offset * -1;
    }

    /**
     * @brief returns the maximum chunk ids
     */
    BaseVector<int> getChunkMaxChunkIndex() const
    {
        BaseVector<int> maxChunkIndex(m_chunkAmount.x - m_chunkIndexOffset.x,
                                      m_chunkAmount.y - m_chunkIndexOffset.y,
                                      m_chunkAmount.z - m_chunkIndexOffset.z);
        return maxChunkIndex;
    }

    /**
     * @brief sets the bounding box in this container and in persistend storage
     *
     * @param boundingBox new bounding box
     */
    void setBoundingBox(const BoundingBox<BaseVector<float>> boundingBox);

  protected:
    /**
     * @brief regenerates cache hash grid
     *
     * @param oldChunkAmount previous amount of chunks
     * @param oldChunkIndexOffset previous offset for chunk indices
     */
    void rehashCache(const BaseVector<std::size_t>& oldChunkAmount,
                     const BaseVector<std::size_t>& oldChunkIndexOffset);

    void expandBoundingBox(const val_type& data);

    /**
     * @brief loads a chunk from persistent storage into cache
     *
     * @tparam T Type of chunk data
     * @param layer layer of chunk
     * @param x x coordinate of chunk in chunk coordinates
     * @param y y coordinate of chunk in chunk coordinates
     * @param z z coordinate of chunk in chunk coordinates
     *
     * @return true if chunk has been loaded; false if chunk does not exist in persistend storage
     */
    template <typename T>
    bool loadChunk(std::string layer, int x, int y, int z);

    /**
     * @brief loads given chunk data into cache
     *
     * Loads given chunk data into cache and handles cache overflows.
     * If the cache is full after adding the chunk, the least recent used chunk will be removed from
     * cache.
     *
     * @param layer layer of chunk
     * @param x x coordinate of chunk in chunk coordinates
     * @param y y coordinate of chunk in chunk coordinates
     * @param z z coordinate of chunk in chunk coordinates
     * @param data content of chunk to load
     */
    void loadChunk(std::string layer, int x, int y, int z, const val_type& data);

    /**
     * @brief sets chunk size in this container and in persistent storage
     *
     * @param chunkSize new size of chunks
     */
    void setChunkSize(float chunkSize)
    {
        m_chunkSize = chunkSize;
        m_io.saveChunkSize(m_chunkSize);
    }

    // bounding box of the entire chunked model
    // i had to make this protected because of the getGlobalBoundingBox() function in ChunkManager
    BoundingBox<BaseVector<float>> m_boundingBox;

  private:
    /**
     * @brief sets the amount of chunks in x y and z direction in this container and in persistent
     * storage
     *
     * @param chunkAmount new amounts of chunks
     */
    void setChunkAmountAndOffset(const BaseVector<std::size_t>& chunkAmount,
                                 const BaseVector<std::size_t>& chunkIndexOffset);

  private:
    // chunkIO for the HDF5 file-IO
    io m_io;

    // number of chunks that will be cached before deleting old chunks
    size_t m_cacheSize;

    // ordered list to save recently used hashValues for the lru cache
    std::list<std::pair<std::string, size_t>> m_items;

    // hash map containing chunked meshes
    std::unordered_map<std::string, std::unordered_map<size_t, val_type>> m_hashGrid;

    // size of chunks
    float m_chunkSize;

    // amount of chunks
    BaseVector<std::size_t> m_chunkAmount;

    // offset of chunks to make chunk index start at 0
    BaseVector<std::size_t> m_chunkIndexOffset;
};

} /* namespace lvr2 */

#include "ChunkHashGrid.tcc"

#endif // CHUNK_HASH_GRID_HPP
