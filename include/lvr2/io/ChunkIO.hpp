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
* ChunkIO.hpp
*
* @date 05.09.2019
* @author Raphael Marx
*/

#ifndef CHUNKIO_HPP
#define CHUNKIO_HPP

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/Model.hpp"

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/hdf5/ArrayIO.hpp"
#include "lvr2/io/hdf5/ChannelIO.hpp"
#include "lvr2/io/hdf5/MeshIO.hpp"
#include "lvr2/io/hdf5/VariantChannelIO.hpp"

namespace lvr2 {
using ChunkHDF5IO = lvr2::Hdf5IO<
        lvr2::hdf5features::ArrayIO,
        lvr2::hdf5features::ChannelIO,
        lvr2::hdf5features::VariantChannelIO,
        lvr2::hdf5features::MeshIO>;

class ChunkIO{

    public:
        ChunkIO();
        /**
         * @brief ChunkIO used for saving and loading chunks into/from a HDF5-file
         *
         */
        ChunkIO(std::string filePath);

        /**
         * @brief write amount, chunksize and the bounding box in the hdf5 file
         */
        void writeBasicStructure(BaseVector<std::size_t> amount, float chunksize, BoundingBox<BaseVector<float>> boundingBox);

        /**
        * @brief write a mesh in a group with the given cellIndex
        */
        void writeChunk(lvr2::MeshBufferPtr mesh, size_t x, size_t y, size_t z);

        /**
        * @brief load a mesh from a group with the given cellIndex
        */
        lvr2::MeshBufferPtr loadChunk(std::string chunkName);

        /**
         * @brief loads and returns a BaseVector with the amount of chunks in each dimension
         */
        BaseVector <size_t> loadAmount();

        /**
         * @brief loads and returns the bounding box for the complete original mesh (that got chunked)
         */
        BoundingBox <BaseVector<float>> loadBoundingBox();

        /**
         * @brief loads and returns the chunksize
         */
        float loadChunkSize();
        /**
         * @brief save reconstruction voxelsize in HDF5
         */
        void writeVoxelSize(float voxelSize);
        /**
         * @brief returns the voxelsize
         */
        float readVoxelSize();
        /**
         * @brief load the extruded value for the cells/voxel of one chunk
         */
        boost::shared_array<bool> lsrLoadExtruded(string chunkName);
        /**
         * @brief loads and returns the Querypoint-distances for the cells/voxel of one chunk
         */
        boost::shared_array<float> lsrLoadQueryPoints(string chunkName);
        /**
         * @brief loads and returns the number of cells/voxel in the chunk
         */
        size_t lsrLoadNumCells(string chunkName);
        /**
         * @brief loads and returns the centers for the cells/voxel in one chunk
         */
        boost::shared_array<float> lsrLoadCenters(string chunkName);
        /**
         * @brief Writes the parameters in the HDF5 file.
         * @param chunkName name of the chunk (typically the grid coordinates)
         * @param csize number of cells/voxel in the chunk
         * @param centers center values for the voxel
         * @param extruded extruded values for the voxel
         * @param queryPoints Querypoint distances for the voxel
         */
        void writeTSDF(string cellName,
                        size_t csize,
                        boost::shared_array<float> centers,
                        boost::shared_array<bool> extruded,
                        boost::shared_array<float> queryPoints);

    private:
        // path to the hdf5 file
        std::string m_filePath;
        // hdf5Io object to save and load chunks
        ChunkHDF5IO m_hdf5IO;


        // default names for the groups
        const std::string m_chunkName = "chunks";
        const std::string m_amountName = "amount";
        const std::string m_chunkSizeName = "size";
        const std::string m_boundingBoxName = "bounding_box";


};
} // namespace lvr2

#endif //CHUNKIO_HPP
