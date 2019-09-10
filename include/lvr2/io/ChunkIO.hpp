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
//#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/io/Model.hpp"
#include "lvr2/io/Hdf5IO.hpp"
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
        /**
         * @brief ChunkIO used for saving and loading chunks into/from a HDF5-file
         *
         */
        ChunkIO(std::string filePath);

        //~ChunkIO();

        void writeBasicStructure(BaseVector<std::size_t> amount, float chunksize, BoundingBox<BaseVector<float>> boundingBox);
        void writeChunk(lvr2::MeshBufferPtr mesh, size_t x, size_t y, size_t z);

    private:
        std::string m_filePath;
        ChunkHDF5IO m_hdf5IO;


};
} // namespace lvr2

#endif //CHUNKIO_HPP
