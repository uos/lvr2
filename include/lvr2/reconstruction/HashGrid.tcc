/**
 * Copyright (c) 2018, University Osnabrück
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

/*
 * HashGrid.cpp
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/io/ChunkIO.hpp"
#include "lvr2/util/Progress.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/reconstruction/FastReconstructionTables.hpp"
#include "lvr2/reconstruction/HashGrid.hpp"

#include <fstream>
#include <iostream>

namespace lvr2
{

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(float cellSize,
                                   BoundingBox<BaseVecT> boundingBox,
                                   bool isVoxelsize,
                                   bool extrude)
    : GridBase(extrude), m_boundingBox(boundingBox)
{
    auto newMax = m_boundingBox.getMax();
    auto newMin = m_boundingBox.getMin();
    if (m_boundingBox.getXSize() < 3 * cellSize)
    {
        newMax.x += cellSize;
        newMin.x -= cellSize;
    }
    if (m_boundingBox.getYSize() < 3 * cellSize)
    {
        newMax.y += cellSize;
        newMin.y -= cellSize;
    }
    if (m_boundingBox.getZSize() < 3 * cellSize)
    {
        newMax.z += cellSize;
        newMin.z -= cellSize;
    }
    m_boundingBox.expand(newMax);
    m_boundingBox.expand(newMin);

    if (!m_boundingBox.isValid())
    {
        cout << timestamp << "Warning: Malformed BoundingBox." << endl;
    }

    if (!isVoxelsize)
    {
        m_voxelsize = (float)m_boundingBox.getLongestSide() / cellSize;
        cout << timestamp << "Used voxelsize is " << m_voxelsize << endl;
    }
    else
    {
        m_voxelsize = cellSize;
    }

    if (!m_extrude)
    {
        cout << timestamp << "Grid is not extruded." << endl;
    }

    BoxT::m_voxelsize = m_voxelsize;
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(string file)
{
    ifstream ifs(file.c_str());
    float minx, miny, minz, maxx, maxy, maxz, vsize;
    size_t qsize, csize;

    ifs >> m_extrude;
    m_extrude = false;
    ifs >> minx >> miny >> minz >> maxx >> maxy >> maxz >> qsize >> vsize >> csize;

    m_boundingBox = BoundingBox<BaseVecT>(BaseVecT(minx, miny, minz), BaseVecT(maxx, maxy, maxz));
    m_voxelsize = vsize;
    BoxT::m_voxelsize = m_voxelsize;

    float pdist;
    BaseVecT v;
    // cout << timestamp << "Creating Grid..." << endl;

    // Iterator over all points, calc lattice indices and add lattice points to the grid
    for (size_t i = 0; i < qsize; i++)
    {

        ifs >> v.x >> v.y >> v.z >> pdist;

        QueryPoint<BaseVecT> qp(v, pdist);
        m_queryPoints.push_back(qp);
    }
    // cout << timestamp << "read qpoints.. csize: " << csize << endl;
    Vector3i index;
    uint cell[8];
    BaseVecT cell_center;
    bool fusion = false;
    for (size_t k = 0; k < csize; k++)
    {
        // cout << "i: " << k << endl;
        ifs >> index.x() >> index.y() >> index.z() >> cell[0] >> cell[1] >> cell[2] >> cell[3] >> cell[4] >> cell[5] >> cell[6] >>
            cell[7] >> cell_center.x >> cell_center.y >> cell_center.z >> fusion;
        BoxT* box = new BoxT(cell_center);
        box->m_extruded = fusion;
        for (int j = 0; j < 8; j++)
        {
            box->setVertex(j, cell[j]);
        }

        m_cells[index] = box;
    }
    cout << timestamp << "Reading cells.." << endl;

    cout << "c size: " << m_cells.size() << endl;
    fillNeighbors();
    cout << "Finished reading grid" << endl;
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(std::vector<string>& files,
                                   BoundingBox<BaseVecT>& boundingBox,
                                   float voxelsize)
    : m_boundingBox(boundingBox), m_voxelsize(voxelsize)
{
    float distances[8];
    BaseVecT box_center;
    Vector3i index;
    bool extruded;
    for (int numFiles = 0; numFiles < files.size(); numFiles++)
    {
        cout << "Loading grid: " << numFiles << "/" << files.size() << endl;

        FILE* pFile = fopen(files[numFiles].c_str(), "rb");
        size_t numCells;
        size_t r = fread(&numCells, sizeof(size_t), 1, pFile);

        for (size_t cellCount = 0; cellCount < numCells; cellCount++)
        {
            r = fread(&(box_center[0]), sizeof(float), 1, pFile);
            r = fread(&(box_center[1]), sizeof(float), 1, pFile);
            r = fread(&(box_center[2]), sizeof(float), 1, pFile);

            r = fread(&extruded, sizeof(bool), 1, pFile);

            r = fread(&(distances[0]), sizeof(float), 8, pFile);

            calcIndex(box_center, index);
            if (this->m_cells.find(index) == this->m_cells.end() && !extruded)
            {
                addBox(index, box_center, distances);
            }
        }
        fclose(pFile);
    }

    fillNeighbors();
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(std::vector<string>& files,
                                   std::vector<BoundingBox<BaseVecT>> innerBoxes,
                                   BoundingBox<BaseVecT>& boundingBox,
                                   float voxelsize)
        : m_boundingBox(boundingBox), m_voxelsize(voxelsize)
{
    float distances[8];
    BaseVecT box_center;
    Vector3i index;
    bool extruded;
    for (int numFiles = 0; numFiles < files.size(); numFiles++)
    {
        // get the min and max vector of the inner chunk bounding box
        BaseVecT innerChunkMin = innerBoxes.at(numFiles).getMin();
        BaseVecT innerChunkMax = innerBoxes.at(numFiles).getMax();

        unsigned int current_index = 0;
        cout << "Loading grid: " << numFiles << "/" << files.size() << endl;

        FILE* pFile = fopen(files[numFiles].c_str(), "rb");
        size_t numCells;
        size_t r = fread(&numCells, sizeof(size_t), 1, pFile);

        for (size_t cellCount = 0; cellCount < numCells; cellCount++)
        {
            r = fread(&(box_center[0]), sizeof(float), 1, pFile);
            r = fread(&(box_center[1]), sizeof(float), 1, pFile);
            r = fread(&(box_center[2]), sizeof(float), 1, pFile);

            r = fread(&extruded, sizeof(bool), 1, pFile);

            r = fread(&(distances[0]), sizeof(float), 8, pFile);

            // Check if the voxel is inside of our bounding box.
            // If not, we skip it, because some other chunk is responsible for the voxel.
            if(box_center.x < innerChunkMin.x || box_center.y < innerChunkMin.y || box_center.z < innerChunkMin.z ||
                    box_center.x > innerChunkMax.x || box_center.y > innerChunkMax.y || box_center.z > innerChunkMax.z )
            {
                continue;
            }

            calcIndex(box_center, index);
            if (this->m_cells.find(index) == this->m_cells.end() && !extruded)
            {
                addBox(index, box_center, distances);
            }
        }
        fclose(pFile);
    }

    fillNeighbors();
}

template<typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(std::vector<PointBufferPtr> chunks,
                                   std::vector<BoundingBox<BaseVecT>> innerBoxes,
                                   BoundingBox<BaseVecT>& boundingBox,
                                   float voxelSize)
        : m_boundingBox(boundingBox)
{
    unsigned int INVALID = BoxT::INVALID_INDEX;
    m_voxelsize = voxelSize;
    size_t numCells;
    boost::shared_array<float> centers;
    boost::shared_array<float> queryPoints;
    boost::shared_array<bool> extruded;
    Vector3i index;
    std::cout << timestamp.getElapsedTime() << "Number of Chunks: "<< chunks.size()<< std::endl;
    std::string comment = timestamp.getElapsedTime() + "Loading grid ";
    lvr2::ProgressBar progress(chunks.size(), comment);
    for (size_t i = 0; i < chunks.size(); i++)
    {
        auto& chunk = chunks.at(i);
        boost::optional<unsigned int> optNumCells =  chunk->getAtomic<unsigned int>("num_voxel");
        boost::optional<Channel<bool>> optExtruded = chunk->getChannel<bool>("extruded");
        boost::optional<Channel<float>> optTSDF = chunk->getFloatChannel("tsdf_values");
        centers = chunk->getPointArray();

        if(optNumCells && optExtruded && optTSDF)
        {
            queryPoints = optTSDF.get().dataPtr();
            extruded = optExtruded.get().dataPtr();
            numCells = optNumCells.get();

            // get the min and max vector of the inner chunk bounding box
            BaseVecT innerChunkMin = innerBoxes.at(i).getMin();
            BaseVecT innerChunkMax = innerBoxes.at(i).getMax();

            for(size_t cellCount = 0; cellCount < numCells; cellCount++)
            {
                // Check if the voxel is inside of our bounding box.
                // If not, we skip it, because some other chunk is responsible for the voxel.
                BaseVecT voxCenter = BaseVecT(centers[cellCount * 3 + 0], centers[cellCount * 3 + 1], centers[cellCount * 3 + 2]);
                if(voxCenter.x < innerChunkMin.x || voxCenter.y < innerChunkMin.y || voxCenter.z < innerChunkMin.z ||
                    voxCenter.x > innerChunkMax.x || voxCenter.y > innerChunkMax.y || voxCenter.z > innerChunkMax.z )
                {
                    continue;
                }

                calcIndex(voxCenter, index);
                if (m_cells.find(index) == m_cells.end() && !extruded[cellCount])
                {
                    float* distances = queryPoints.get() + cellCount * 8;
                    addBox(index, voxCenter, distances);
                }
            }
        }
        else
        {
            std::cout << "WARNING: something went wrong while reconstructing multiple chunks. Please check if all channels are available." << std::endl;
        }
        if(!timestamp.isQuiet())
            ++progress;
    }
    if(!timestamp.isQuiet())
        cout << endl;

    fillNeighbors();
}

template <typename BaseVecT, typename BoxT>
PointBufferPtr HashGrid<BaseVecT, BoxT>::toPointBuffer() const
{
    size_t n = m_cells.size();
    boost::shared_array<float> centers(new float[3 * n]);
    boost::shared_array<bool> extruded(new bool[n]);
    boost::shared_array<float> queryPoints(new float[8 * n]);

    std::vector<size_t> bucketOffsets(m_cells.bucket_count());
    bucketOffsets[0] = 0;
    for (size_t i = 1; i < m_cells.bucket_count(); i++)
    {
        bucketOffsets[i] = bucketOffsets[i - 1] + m_cells.bucket_size(i - 1);
    }

    #pragma omp parallel for schedule(dynamic,32)
    for (size_t i = 0; i < m_cells.bucket_count(); i++)
    {
        size_t offset = bucketOffsets[i];
        auto start = m_cells.begin(i), end = m_cells.end(i);
        for (auto it = start; it != end; ++it)
        {
            auto& [ index, cell ] = *it;
            for (int j = 0; j < 3; ++j)
            {
                centers[3 * offset + j] = cell->getCenter()[j];
            }

            extruded[offset] = cell->m_extruded;

            for (int k = 0; k < 8; ++k)
            {
                queryPoints[8 * offset + k] = m_queryPoints[cell->getVertex(k)].m_distance;
            }
            offset++;
        }
    }

    auto ret = std::make_shared<PointBuffer>(centers, n);
    ret->addFloatChannel(queryPoints, "tsdf_values", n, 8);
    ret->addChannel(extruded, "extruded", n, 1);
    ret->addAtomic<unsigned int>(n, "num_voxel");
    return ret;
}

template <typename BaseVecT, typename BoxT>
BoxT* HashGrid<BaseVecT, BoxT>::addBox(const Vector3i& index, const BaseVecT& center, float* distances)
{
    BoxT* box = new BoxT(center);
    uint current_index;
    BaseVecT offset, position;
    for (int i = 0; i < 8; i++)
    {
        current_index = this->findQueryPoint(i, index);
        if (current_index == BoxT::INVALID_INDEX)
        {
            current_index = this->m_queryPoints.size();
            offset = BaseVecT(box_creation_table[i][0], box_creation_table[i][1], box_creation_table[i][2]);
            position = center + offset * (m_voxelsize / 2.0f);
            this->m_queryPoints.push_back(QueryPoint<BaseVecT>(position, distances[i]));
        }
        box->setVertex(i, current_index);
    }
    if (!m_cells.emplace(index, box).second)
    {
        delete box;
        throw std::runtime_error("HashGrid::addBox: Cell already exists!");
    }
    return box;
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::fillNeighbors()
{
    #pragma omp parallel for schedule(dynamic,12)
    for (size_t i = 0; i < m_cells.bucket_count(); i++)
    {
        int neighbor_index;
        typename box_map::iterator neighbor_it;
        auto start = m_cells.begin(i), end = m_cells.end(i);
        for (auto it = start; it != end; ++it)
        {
            auto& [ index, cell ] = *it;
            neighbor_index = 0;
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        neighbor_it = this->m_cells.find(index + Vector3i(dx, dy, dz));

                        // If it exists, save pointer in box
                        if (neighbor_it != this->m_cells.end())
                        {
                            cell->setNeighbor(neighbor_index, neighbor_it->second);
                            neighbor_it->second->setNeighbor(26 - neighbor_index, cell); // TODO: should be redundant: neighbor will find this box
                        }

                        neighbor_index++;
                    }
                }
            }
        }
    }
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::addLatticePoint(int i, int j, int k, float distance)
{
    Vector3i index(i, j, k);
    if (m_cells.find(index) != m_cells.end())
    {
        return;
    }

    float distances[8];
    std::fill_n(distances, 8, distance);
    auto cell = addBox(index, distances);

    int neighbor_index = 0;
    typename box_map::iterator neighbor_it;
    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dz = -1; dz <= 1; dz++)
            {
                neighbor_it = this->m_cells.find(index + Vector3i(dx, dy, dz));

                // If it exists, save pointer in box
                if (neighbor_it != this->m_cells.end())
                {
                    cell->setNeighbor(neighbor_index, neighbor_it->second);
                    neighbor_it->second->setNeighbor(26 - neighbor_index, cell);
                }

                neighbor_index++;
            }
        }
    }
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::addLatticePoints(const std::unordered_set<Vector3i>& points)
{
    m_cells.reserve(points.size());

    float distances[8] = { 0.0f };
    for (auto& index : points)
    {
        addBox(index, distances);
    }

    fillNeighbors();
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::~HashGrid()
{
    for (auto& [ _, cell ] : m_cells)
    {
        delete cell;
    }

    m_cells.clear();
}

template <typename BaseVecT, typename BoxT>
uint HashGrid<BaseVecT, BoxT>::findQueryPoint(int position, const Vector3i& index) const
{
    const int* table = shared_vertex_table[position];
    const int* table_end = table + 28;

    for (const int* t = table; t < table_end; t += 4)
    {
        auto it = m_cells.find(index + Vector3i(t[0], t[1], t[2]));
        if (it != m_cells.end())
        {
            uint index = it->second->getVertex(t[3]);
            if (index != BoxT::INVALID_INDEX)
            {
                return index;
            }
        }
    }

    return BoxT::INVALID_INDEX;
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::saveGrid(string filename)
{
    std::cout << timestamp << "Writing grid..." << std::endl;

    // Open file for writing
    std::ofstream out(filename.c_str());

    // Write data
    if (out.good())
    {
        // Write header
        out << m_queryPoints.size() << " " << m_voxelsize << " " << m_cells.size() << endl;

        // Write query points and distances
        for (size_t i = 0; i < m_queryPoints.size(); i++)
        {
            out << m_queryPoints[i].m_position.x << " " << m_queryPoints[i].m_position.y << " "
                << m_queryPoints[i].m_position.z << " ";

            if (!isnan(m_queryPoints[i].m_distance))
            {
                out << m_queryPoints[i].m_distance << std::endl;
            }
            else
            {
                out << 0 << std::endl;
            }
        }

        // Write box definitions
        for (auto& [ _, cell ] : m_cells)
        {
            for (int i = 0; i < 8; i++)
            {
                out << cell->getVertex(i) << " ";
            }
            out << std::endl;
        }
    }
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::saveCells(string file)
{
    FILE* pFile = fopen(file.c_str(), "wb");
    size_t csize = m_cells.size();
    fwrite(&csize, sizeof(size_t), 1, pFile);
    for (auto& [ _, cell ] : m_cells)
    {
        fwrite(&cell->getCenter()[0], sizeof(float), 1, pFile);
        fwrite(&cell->getCenter()[1], sizeof(float), 1, pFile);
        fwrite(&cell->getCenter()[2], sizeof(float), 1, pFile);

        fwrite(&cell->m_extruded, sizeof(bool), 1, pFile);

        fwrite(&m_queryPoints[cell->getVertex(0)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[cell->getVertex(1)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[cell->getVertex(2)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[cell->getVertex(3)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[cell->getVertex(4)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[cell->getVertex(5)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[cell->getVertex(6)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[cell->getVertex(7)].m_distance, sizeof(float), 1, pFile);
    }
    fclose(pFile);
}
// <<<<<<< HEAD
// =======
// template <typename BaseVecT, typename BoxT>
// void HashGrid<BaseVecT, BoxT>::saveCellsHDF5(string file, string groupName)
// {
//     lvr2::ChunkIO chunkIo = ChunkIO(file);

//     chunkIo.writeVoxelSize(m_voxelsize);
//     size_t csize = getNumberOfCells();

//     boost::shared_array<float> centers(new float[3 * csize]);
//     boost::shared_array<bool> extruded(new bool[csize]);
//     boost::shared_array<float> queryPoints(new float[8 * csize]);

//     int counter = 0;
//     for (auto it = firstCell(); it != lastCell(); it++)
//     {
//         for (int j = 0; j < 3; ++j)
//         {
//             centers[3 * counter + j] = it->second->getCenter()[j];
//         }

//         extruded[counter] = it->second->m_extruded;

//         for (int k = 0; k < 8; ++k)
//         {
//             queryPoints[8 * counter + k] = m_queryPoints[it->second->getVertex(k)].m_distance;
//         }
//         ++counter;
//     }
//     chunkIo.writeTSDF(groupName, csize, centers, extruded, queryPoints);
// }
// >>>>>>> feature/scan_project_io_fix

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::serialize(string file)
{
    std::cout << timestamp << "saving grid: " << file << std::endl;
    std::ofstream out(file.c_str());

    // Write data
    if (out.good())
    {
        out << m_extrude << std::endl;
        out << m_boundingBox.getMin().x << " " << m_boundingBox.getMin().y << " "
            << m_boundingBox.getMin().z << " " << m_boundingBox.getMax().x << " "
            << m_boundingBox.getMax().y << " " << m_boundingBox.getMax().z << std::endl;

        out << m_queryPoints.size() << " " << m_voxelsize << " " << m_cells.size() << endl;

        // Write query points and distances
        for (size_t i = 0; i < m_queryPoints.size(); i++)
        {
            out << m_queryPoints[i].m_position.x << " " << m_queryPoints[i].m_position.y << " "
                << m_queryPoints[i].m_position.z << " ";

            if (!isnan(m_queryPoints[i].m_distance))
            {
                out << m_queryPoints[i].m_distance << std::endl;
            }
            else
            {
                out << 0 << endl;
            }
        }

        // Write box definitions
        for (auto& [ index, cell ] : m_cells)
        {
            out << index.x() << " " << index.y() << " " << index.z() << " ";
            for (int i = 0; i < 8; i++)
            {
                out << cell->getVertex(i) << " ";
            }
            out << cell->getCenter().x << " " << cell->getCenter().y << " " << cell->getCenter().z
                << " " << cell->m_extruded << endl;
        }
    }
    out.close();
    std::cout << timestamp << "finished saving grid: " << file << std::endl;
}

} // namespace lvr2
