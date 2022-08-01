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
HashGrid<BaseVecT, BoxT>::HashGrid(float resolution, BoundingBox<BaseVecT> boundingBox, bool isVoxelsize, bool extrude)
    : GridBase(extrude), m_boundingBox(boundingBox)
{
    m_voxelsize = isVoxelsize ? resolution : (m_boundingBox.getLongestSide() / resolution);

    auto newMax = m_boundingBox.getMax();
    auto newMin = m_boundingBox.getMin();
    if (m_boundingBox.getXSize() < 3 * m_voxelsize)
    {
        newMax.x += m_voxelsize;
        newMin.x -= m_voxelsize;
    }
    if (m_boundingBox.getYSize() < 3 * m_voxelsize)
    {
        newMax.y += m_voxelsize;
        newMin.y -= m_voxelsize;
    }
    if (m_boundingBox.getZSize() < 3 * m_voxelsize)
    {
        newMax.z += m_voxelsize;
        newMin.z -= m_voxelsize;
    }
    m_boundingBox.expand(newMax);
    m_boundingBox.expand(newMin);

    if (!m_boundingBox.isValid())
    {
        cout << timestamp << "Warning: Malformed BoundingBox." << endl;
    }

    BoxT::m_voxelsize = m_voxelsize;
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(const std::vector<std::string>& files,
                                   const std::vector<BoundingBox<BaseVecT>>& innerBoxes,
                                   const BoundingBox<BaseVecT>& boundingBox,
                                   float voxelsize)
        : m_boundingBox(boundingBox), m_voxelsize(voxelsize)
{
    float distances[8];
    BaseVecT box_center;
    Vector3i index;
    BaseVecT innerChunkMin = boundingBox.getMin();
    BaseVecT innerChunkMax = boundingBox.getMax();
    for (int numFiles = 0; numFiles < files.size(); numFiles++)
    {
        if (!innerBoxes.empty())
        {
            // get the min and max vector of the inner chunk bounding box
            innerChunkMin = innerBoxes.at(numFiles).getMin();
            innerChunkMax = innerBoxes.at(numFiles).getMax();
        }

        std::ifstream in(files.at(numFiles), std::ios::in | std::ios::binary);

        unsigned long numCells;
        in >> numCells;

        for (size_t cellCount = 0; cellCount < numCells; cellCount++)
        {
            in >> box_center[0] >> box_center[1] >> box_center[2];
            for (size_t i = 0; i < 8; i++)
            {
                in >> distances[i];
            }

            // Check if the voxel is inside of our bounding box.
            // If not, we skip it, because some other chunk is responsible for the voxel.
            if(box_center.x < innerChunkMin.x || box_center.y < innerChunkMin.y || box_center.z < innerChunkMin.z ||
                    box_center.x > innerChunkMax.x || box_center.y > innerChunkMax.y || box_center.z > innerChunkMax.z )
            {
                continue;
            }

            calcIndex(box_center, index);
            if (this->m_cells.find(index) == this->m_cells.end())
            {
                addBox(index, box_center, distances);
            }
        }
    }

    fillNeighbors();
}

template<typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(const std::vector<PointBufferPtr>& chunks,
                                   const std::vector<BoundingBox<BaseVecT>>& innerBoxes,
                                   const BoundingBox<BaseVecT>& boundingBox,
                                   float voxelsize)
        : m_boundingBox(boundingBox), m_voxelsize(voxelsize)
{
    BaseVecT box_center;
    Vector3i index;
    BaseVecT innerChunkMin = boundingBox.getMin();
    BaseVecT innerChunkMax = boundingBox.getMax();
    std::unique_ptr<ProgressBar> progress = nullptr;
    if (chunks.size() > 1)
    {
        std::cout << timestamp.getElapsedTime() << "Number of Chunks: "<< chunks.size()<< std::endl;
        std::string comment = timestamp.getElapsedTime() + "Loading grid ";
        progress.reset(new ProgressBar(chunks.size(), comment));
    }
    for (size_t i = 0; i < chunks.size(); i++)
    {
        auto& chunk = chunks.at(i);
        size_t numCells = chunk->numPoints();
        auto centers = *chunk->getFloatChannel("points");
        auto queryPoints = chunk->getFloatChannel("tsdf_values")->dataPtr();

        if (!innerBoxes.empty())
        {
            // get the min and max vector of the inner chunk bounding box
            innerChunkMin = innerBoxes.at(i).getMin();
            innerChunkMax = innerBoxes.at(i).getMax();
        }

        for(size_t cellCount = 0; cellCount < numCells; cellCount++)
        {
            // Check if the voxel is inside of our bounding box.
            // If not, we skip it, because some other chunk is responsible for the voxel.
            box_center = centers[cellCount];
            if(box_center.x < innerChunkMin.x || box_center.y < innerChunkMin.y || box_center.z < innerChunkMin.z ||
                box_center.x > innerChunkMax.x || box_center.y > innerChunkMax.y || box_center.z > innerChunkMax.z )
            {
                continue;
            }

            calcIndex(box_center, index);
            if (m_cells.find(index) == m_cells.end())
            {
                float* distances = queryPoints.get() + cellCount * 8;
                addBox(index, box_center, distances);
            }
        }

        if(progress)
            ++(*progress);
    }
    if(progress)
        std::cout << std::endl;

    fillNeighbors();
}

template <typename BaseVecT, typename BoxT>
PointBufferPtr HashGrid<BaseVecT, BoxT>::toPointBuffer() const
{
    size_t n = m_cells.size();
    boost::shared_array<float> centers(new float[3 * n]);
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
            for (int k = 0; k < 8; ++k)
            {
                queryPoints[8 * offset + k] = m_queryPoints[cell->getVertex(k)].m_distance;
            }
            offset++;
        }
    }

    auto ret = std::make_shared<PointBuffer>(centers, n);
    ret->addFloatChannel(queryPoints, "tsdf_values", n, 8);
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
void HashGrid<BaseVecT, BoxT>::saveGrid(std::string file)
{
    std::ofstream out(file, std::ios::out | std::ios::binary);

    unsigned long csize = m_cells.size();
    out << csize;
    for (auto& [ _, cell ] : m_cells)
    {
        auto& center = cell->getCenter();
        out << center[0] << center[1] << center[2];

        for (size_t i = 0; i < 8; i++)
        {
            out << m_queryPoints[cell->getVertex(i)].m_distance;
        }
    }
}

} // namespace lvr2
