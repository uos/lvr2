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

#include <iostream>
#include <string>

#include "lvr2/io/ChunkIO.hpp"
#include "lvr2/io/hdf5/ArrayIO.hpp"



namespace lvr2 {

ChunkIO::ChunkIO(std::string filePath) :m_filePath(filePath)
{
    m_hdf5IO.open(m_filePath);
}

void ChunkIO::writeBasicStructure(BaseVector<std::size_t> amount, float chunksize, BoundingBox<BaseVector<float>> boundingBox)
{
    // HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, "chunks", true); // <- maybe use this instead of string

    boost::shared_array<size_t> amountArr(new size_t[3] {amount.x, amount.y, amount.z});
    m_hdf5IO.save("chunks", "amount", 3, amountArr);
    boost::shared_array<float> chunkSizeArr(new float[1] {chunksize});
    m_hdf5IO.save("chunks", "size", 1, chunkSizeArr);
    boost::shared_array<float> boundingBoxArr(
            new float[6] {boundingBox.getMin()[0], boundingBox.getMin()[1], boundingBox.getMin()[2],
                            boundingBox.getMax()[0], boundingBox.getMax()[1], boundingBox.getMax()[2]});
    std::vector<size_t> boundingBoxDim({2, 3});
    m_hdf5IO.save("chunks", "bounding_box", boundingBoxDim, boundingBoxArr);

}
void ChunkIO::writeChunk(lvr2::MeshBufferPtr mesh, size_t x, size_t y, size_t z)
{
    HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, "chunks", true);
    std::string gridCoord = std::to_string(x) + "-" + std::to_string(y) + "-" + std::to_string(z);
    if (!chunks.exist(gridCoord))
    {
        chunks.createGroup(gridCoord);
    }
    HighFive::Group meshGroup = chunks.getGroup(gridCoord);

    m_hdf5IO.save(meshGroup, mesh);
}


} // namespace lvr2