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
* ChunkIO.cpp
*
* @date 05.09.2019
* @author Raphael Marx
*/

#include <iostream>
#include <string>

#include "lvr2/io/ChunkIO.hpp"



namespace lvr2 {

ChunkIO::ChunkIO()
= default;

ChunkIO::ChunkIO(std::string filePath)
:m_filePath(filePath)
{
    m_hdf5IO.open(m_filePath);
}

void ChunkIO::writeBasicStructure(BaseVector<std::size_t> amount, float chunksize, BoundingBox<BaseVector<float>> boundingBox)
{
    // HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, "chunks", true); // <- maybe use this instead of string

    boost::shared_array<size_t> amountArr(new size_t[3] {amount.x, amount.y, amount.z});
    m_hdf5IO.save(m_chunkName, m_amountName, 3, amountArr);

    boost::shared_array<float> chunkSizeArr(new float[1] {chunksize});
    m_hdf5IO.save(m_chunkName, m_chunkSizeName, 1, chunkSizeArr);

    boost::shared_array<float> boundingBoxArr(
            new float[6] {boundingBox.getMin()[0], boundingBox.getMin()[1], boundingBox.getMin()[2],
                            boundingBox.getMax()[0], boundingBox.getMax()[1], boundingBox.getMax()[2]});
    std::vector<size_t> boundingBoxDim({2, 3});
    m_hdf5IO.save(m_chunkName, m_boundingBoxName, boundingBoxDim, boundingBoxArr);
}

void ChunkIO::writeChunk(lvr2::MeshBufferPtr mesh, size_t x, size_t y, size_t z)
{
    HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, m_chunkName, true);
    std::string cellName = std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
    if (!chunks.exist(cellName))
    {
        chunks.createGroup(cellName);
    }
    HighFive::Group meshGroup = chunks.getGroup(cellName);

    m_hdf5IO.save(meshGroup, mesh);
}

BaseVector<size_t> ChunkIO::loadAmount()
{
    BaseVector<size_t> amount;
    size_t dimensionAmount;
    boost::shared_array<size_t> amountArr
            = m_hdf5IO.ArrayIO::load<size_t>(m_chunkName, m_amountName, dimensionAmount);
    if(dimensionAmount != 3)
    {
        std::cout   << "Error loading chunk data: amount has not the right dimension. Real: "
                    << dimensionAmount << "; Expected: 3" << std::endl;
    }
    else
    {
        amount = BaseVector<size_t>(amountArr[0], amountArr[1], amountArr[2]);
    }
    return amount;
}

float ChunkIO::loadChunkSize()
{
    float chunkSize;
    size_t dimensionChunkSize;
    boost::shared_array<float> chunkSizeArr
            = m_hdf5IO.ArrayIO::load<float>(m_chunkName, m_chunkSizeName, dimensionChunkSize);
    if(dimensionChunkSize != 1)
    {
        std::cout   << "Error loading chunk data: chunkSize has not the right dimension. Real: "
                    << dimensionChunkSize << "; Expected: 1" << std::endl;
        chunkSize = 0;
    }
    else
    {
        chunkSize = chunkSizeArr[0];
    }
    return chunkSize;
}

BoundingBox<BaseVector<float>> ChunkIO::loadBoundingBox()
{
    BoundingBox<BaseVector<float>> boundingBox;
    std::vector<size_t> dimensionBox;
    boost::shared_array<float> boundingBoxArr
            = m_hdf5IO.ArrayIO::load<float>(m_chunkName, m_boundingBoxName, dimensionBox);
    if(dimensionBox.at(0) != 2 && dimensionBox.at(1) != 3)
    {
        std::cout   << "Error loading chunk data: bounding_box has not the right dimension. Real: {"
                    << dimensionBox.at(0) << ", " << dimensionBox.at(1) << "}; Expected: {2, 3}" << std::endl;
    }
    else
    {
        boundingBox = BoundingBox<BaseVector<float>>(
                BaseVector<float>(boundingBoxArr[0], boundingBoxArr[1], boundingBoxArr[2]),
                BaseVector<float>(boundingBoxArr[3], boundingBoxArr[4], boundingBoxArr[5]));
    }
    return boundingBox;
}

lvr2::MeshBufferPtr ChunkIO::loadChunk(std::string chunkName)
{
    return m_hdf5IO.loadMesh(m_chunkName + "/" + chunkName);
}

void ChunkIO::writeVoxelSize(float voxelSize) {
    HighFive::Group g = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, "", true);

    if (!g.exist(m_chunkName))
    {
        g.createGroup(m_chunkName);
    }

    boost::shared_array<float> voxelSizeArr(new float[1] {voxelSize});
    m_hdf5IO.save(m_chunkName, "voxelsize", 1, voxelSizeArr);
}
float ChunkIO::readVoxelSize()
{
    size_t len;
    boost::shared_array<float> voxelSize = m_hdf5IO.loadArray<float>(m_chunkName, "voxelsize", len);
    return voxelSize.get()[0];
}

void ChunkIO::writeTSDF(string chunkName,
                        size_t csize,
                        boost::shared_array<float> centers,
                        boost::shared_array<bool> extruded,
                        boost::shared_array<float> queryPoints) {
    HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, m_chunkName, true);

    if (!chunks.exist(chunkName))
    {
        chunks.createGroup(chunkName);
    }

    boost::shared_array<size_t> numCellsArr = boost::shared_array<size_t>(new size_t[1]);
    numCellsArr.get()[0] = csize;
    m_hdf5IO.save(m_chunkName + "/" + chunkName, "numCells", 1, numCellsArr);
    std::vector<size_t> centersDim({csize, 3});
    m_hdf5IO.save(m_chunkName + "/" + chunkName, "centers", centersDim, centers);
    m_hdf5IO.save(m_chunkName + "/" + chunkName, "extruded", csize, extruded);
    std::vector<size_t> queryPointsDim({csize, 8});
    m_hdf5IO.save(m_chunkName + "/" + chunkName, "queryPoints", queryPointsDim, queryPoints);
}

boost::shared_array<float> ChunkIO::lsrLoadCenters(std::string chunkName)
{
    HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, m_chunkName, false);

    if (!chunks.exist(chunkName))
    {
        // chunk does not exist
        return boost::shared_array<float>();
    }
    HighFive::Group single_chunk = chunks.getGroup(chunkName);


    std::vector<size_t> dimensionCenters;
    boost::shared_array<float> centersArr
            = m_hdf5IO.ArrayIO::load<float>(single_chunk, "centers", dimensionCenters);
    if (dimensionCenters.at(1) != 3)
    {
        std::cout << "Error loading chunk/tsdf data: centers has not the right dimension. Real: {"
                  << dimensionCenters.at(0) << ", " << dimensionCenters.at(1) << "}; Expected: {N, 3}"
                  << std::endl;
        return boost::shared_array<float>();
    }
    return centersArr;
}
size_t ChunkIO::lsrLoadNumCells(std::string chunkName)
{
    HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, m_chunkName, false);

    if (!chunks.exist(chunkName))
    {
        // chunk does not exist
        return 0;
    }
    HighFive::Group single_chunk = chunks.getGroup(chunkName);


    std::vector<size_t> dimensionNumCells;
    boost::shared_array<float> numCellsArr
            = m_hdf5IO.ArrayIO::load<float>(single_chunk, "numCells", dimensionNumCells);
    return numCellsArr.get()[0];
}
boost::shared_array<float> ChunkIO::lsrLoadQueryPoints(std::string chunkName)
{
    HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, m_chunkName, false);

    if (!chunks.exist(chunkName))
    {
        // chunk does not exist
        return boost::shared_array<float>();
    }
    HighFive::Group single_chunk = chunks.getGroup(chunkName);


    std::vector<size_t> dimensionQueryPoints;
    boost::shared_array<float> qpArr
            = m_hdf5IO.ArrayIO::load<float>(single_chunk, "queryPoints", dimensionQueryPoints);
    if (dimensionQueryPoints.at(1) != 8)
    {
        std::cout << "Error loading chunk/tsdf data: queryPoints has not the right dimension. Real: {"
                  << dimensionQueryPoints.at(0) << ", " << dimensionQueryPoints.at(1) << "}; Expected: {N, 8}"
                  << std::endl;
        return boost::shared_array<float>();
    }
    return qpArr;
}
boost::shared_array<bool> ChunkIO::lsrLoadExtruded(std::string chunkName)
{
    HighFive::Group chunks = hdf5util::getGroup(m_hdf5IO.m_hdf5_file, m_chunkName, false);

    if (!chunks.exist(chunkName))
    {
        // chunk does not exist
        return boost::shared_array<bool>();
    }
    HighFive::Group single_chunk = chunks.getGroup(chunkName);
    std::vector<size_t> dimensionExtruded;
    boost::shared_array<bool> extrudedArr
            = m_hdf5IO.ArrayIO::load<bool>(single_chunk, "extruded", dimensionExtruded);
    return extrudedArr;
}

} // namespace lvr2