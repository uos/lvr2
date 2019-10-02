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

#include <list>
#include <unordered_map>

#include "lvr2/io/ChunkIO.hpp"

namespace lvr2
{
class ChunkHashGrid
{
public:
    ChunkHashGrid() = default;
    /**
     * @brief class to load chunks from an HDF5 file
     *
     * @param hdf5Path path to the HDF5 file
     */
    explicit ChunkHashGrid(std::string hdf5Path, size_t cacheSize);

     /**
      * @brief loads a chunk from the HDF5 file into the hash grid
      *
      * @param hashValue hash-value of the chunk
      * @param x grid coordinate in x-dimension
      * @param y grid coordinate in y-dimension
      * @param z grid coordinate in z-dimension
      * @return true if loading was possible, false if loading was not possible
      *         (for example if the hash-value does not belong to a mesh with vertices)
      */
    bool loadChunk(size_t hashValue, int x, int y, int z);

     /**
      * @brief returns a mesh of a given cellIndex
      *
      * @param hashValue hash-value of the chunk
      * @param x grid coordinate in x-dimension
      * @param y grid coordinate in y-dimension
      * @param z grid coordinate in z-dimension
      * @return the MeshBufferPtr containing the chunk
      */
    MeshBufferPtr findChunk(size_t hashValue, int x, int y, int z);

    MeshBufferPtr findChunkCondition(size_t hashValue, int x, int y, int z, std::string channelName);
private:
    // ordered list to save recently used hashValues
    std::list<size_t> items;
    // hash map containing chunked meshes and the position in the items list
    std::unordered_map<size_t, std::pair<MeshBufferPtr, std::list<size_t>::iterator>> m_hashGrid;

    // chunkIO for the HDF5 file-IO
    std::shared_ptr<lvr2::ChunkIO> m_chunkIO;

    // number of chunks that will be cached before deleting old chunks
    size_t m_cacheSize = 100;

    /**
     * @brief Adds a mesh to the hashmap and deletes the least
     * recently used mesh/chunk, if the size exeeds the m_cacheSize.
     *
     * @param hashValue the value, where the chunk will be saved
     * @param mesh the MeshbufferPtr of the chunk
     */
    void set(size_t hashValue, const MeshBufferPtr& mesh);
    /**
     * @brief Searches the hashmap for the mesh with the given hashValue.
     *
     * @param[in] hashValue hashValue of the mesh
     * @param[out] mesh the mesh/chunk
     * @return true, if the hashmap contains the mesh of that hashValue
     */
    bool get(size_t hashValue, MeshBufferPtr& mesh);

};

} /* namespace lvr2 */
#endif //CHUNK_HASH_GRID_HPP
