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
#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"
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
    : GridBase(extrude), m_boundingBox(boundingBox), m_globalIndex(0)
{
    m_coordinateScales = BaseVecT(1, 1, 1);

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
    }
    else
    {
        m_voxelsize = cellSize;
    }

    cout << timestamp << "Used voxelsize is " << m_voxelsize << endl;

    if (!m_extrude)
    {
        cout << timestamp << "Grid is not extruded." << endl;
    }

    BoxT::m_voxelsize = m_voxelsize;
    calcIndices();
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
    m_globalIndex = 0;
    m_coordinateScales.x = 1.0;
    m_coordinateScales.y = 1.0;
    m_coordinateScales.z = 1.0;
    m_voxelsize = vsize;
    BoxT::m_voxelsize = m_voxelsize;
    calcIndices();

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
    size_t h;
    unsigned int cell[8];
    BaseVecT cell_center;
    bool fusion = false;
    for (size_t k = 0; k < csize; k++)
    {
        // cout << "i: " << k << endl;
        ifs >> h >> cell[0] >> cell[1] >> cell[2] >> cell[3] >> cell[4] >> cell[5] >> cell[6] >>
            cell[7] >> cell_center.x >> cell_center.y >> cell_center.z >> fusion;
        BoxT* box = new BoxT(cell_center);
        box->m_extruded = fusion;
        for (int j = 0; j < 8; j++)
        {
            box->setVertex(j, cell[j]);
        }

        m_cells[h] = box;
    }
    cout << timestamp << "Reading cells.." << endl;
    typename HashGrid<BaseVecT, BoxT>::box_map_it it;
    typename HashGrid<BaseVecT, BoxT>::box_map_it neighbor_it;

    cout << "c size: " << m_cells.size() << endl;
    for (it = m_cells.begin(); it != m_cells.end(); it++)
    {
        // cout << "asdfsdfgsdfgdsfgdfg" << endl;
        BoxT* currentBox = it->second;
        int neighbor_index = 0;
        size_t neighbor_hash = 0;

        for (int a = -1; a < 2; a++)
        {
            for (int b = -1; b < 2; b++)
            {
                for (int c = -1; c < 2; c++)
                {

                    // Calculate hash value for current neighbor cell
                    int idx = calcIndex((it->second->getCenter()[0] - m_boundingBox.getMin()[0]) /
                                        m_voxelsize);
                    int idy = calcIndex((it->second->getCenter()[1] - m_boundingBox.getMin()[1]) /
                                        m_voxelsize);
                    int idz = calcIndex((it->second->getCenter()[2] - m_boundingBox.getMin()[2]) /
                                        m_voxelsize);
                    neighbor_hash = this->hashValue(idx + a, idy + b, idz + c);
                    // cout << "n hash: " << neighbor_hash  << endl;
                    // cout << " id: " << neighbor_index << endl;

                    // Try to find this cell in the grid
                    neighbor_it = this->m_cells.find(neighbor_hash);

                    // If it exists, save pointer in box
                    if (neighbor_it != this->m_cells.end())
                    {
                        currentBox->setNeighbor(neighbor_index, (*neighbor_it).second);
                        (*neighbor_it).second->setNeighbor(26 - neighbor_index, currentBox);
                    }

                    neighbor_index++;
                }
            }
        }
    }
    cout << "Finished reading grid" << endl;
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(std::vector<string>& files,
                                   BoundingBox<BaseVecT>& boundingBox,
                                   float voxelsize)
    : m_boundingBox(boundingBox), m_voxelsize(voxelsize), m_globalIndex(0)
{
    unsigned int INVALID = BoxT::INVALID_INDEX;
    calcIndices();
    float distances[8];
    BaseVecT box_center;
    bool extruded;
    float vsh = 0.5 * this->m_voxelsize;
    for (int numFiles = 0; numFiles < files.size(); numFiles++)
    {
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

            size_t idx = calcIndex((box_center[0] - m_boundingBox.getMin()[0]) / m_voxelsize);
            size_t idy = calcIndex((box_center[1] - m_boundingBox.getMin()[1]) / m_voxelsize);
            size_t idz = calcIndex((box_center[2] - m_boundingBox.getMin()[2]) / m_voxelsize);
            size_t hash = hashValue(idx, idy, idz);
            auto cell_it = this->m_cells.find(hash);
            if (cell_it == this->m_cells.end() && !extruded)
            {
                BoxT* box = new BoxT(box_center);
                for (int i = 0; i < 8; i++)
                {
                    current_index = this->findQueryPoint(i, idx, idy, idz);
                    if (current_index != INVALID)
                        box->setVertex(i, current_index);
                    else
                    {
                        BaseVecT position(box_center[0] + box_creation_table[i][0] * vsh,
                                          box_center[1] + box_creation_table[i][1] * vsh,
                                          box_center[2] + box_creation_table[i][2] * vsh);
                        this->m_queryPoints.push_back(QueryPoint<BaseVecT>(position, distances[i]));
                        box->setVertex(i, this->m_globalIndex);
                        this->m_globalIndex++;
                    }
                }
                // Set pointers to the neighbors of the current box
                int neighbor_index = 0;
                size_t neighbor_hash = 0;

                for (int a = -1; a < 2; a++)
                {
                    for (int b = -1; b < 2; b++)
                    {
                        for (int c = -1; c < 2; c++)
                        {

                            // Calculate hash value for current neighbor cell
                            neighbor_hash = this->hashValue(idx + a, idy + b, idz + c);

                            // Try to find this cell in the grid
                            auto neighbor_it = this->m_cells.find(neighbor_hash);

                            // If it exists, save pointer in box
                            if (neighbor_it != this->m_cells.end())
                            {
                                box->setNeighbor(neighbor_index, (*neighbor_it).second);
                                (*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
                            }

                            neighbor_index++;
                        }
                    }
                }

                this->m_cells[hash] = box;
            }
        }
        fclose(pFile);
    }
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(std::vector<string>& files,
                                   std::vector<BoundingBox<BaseVecT>> innerBoxes,
                                   BoundingBox<BaseVecT>& boundingBox,
                                   float voxelsize)
        : m_boundingBox(boundingBox), m_voxelsize(voxelsize), m_globalIndex(0)
{
    unsigned int INVALID = BoxT::INVALID_INDEX;
    calcIndices();
    float distances[8];
    BaseVecT box_center;
    bool extruded;
    float vsh = 0.5 * this->m_voxelsize;
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

            size_t idx = calcIndex((box_center[0] - m_boundingBox.getMin()[0]) / m_voxelsize);
            size_t idy = calcIndex((box_center[1] - m_boundingBox.getMin()[1]) / m_voxelsize);
            size_t idz = calcIndex((box_center[2] - m_boundingBox.getMin()[2]) / m_voxelsize);
            size_t hash = hashValue(idx, idy, idz);
            auto cell_it = this->m_cells.find(hash);
            if (cell_it == this->m_cells.end() && !extruded)
            {
                BoxT* box = new BoxT(box_center);
                for (int i = 0; i < 8; i++)
                {
                    current_index = this->findQueryPoint(i, idx, idy, idz);
                    if (current_index != INVALID)
                        box->setVertex(i, current_index);
                    else
                    {
                        BaseVecT position(box_center[0] + box_creation_table[i][0] * vsh,
                                          box_center[1] + box_creation_table[i][1] * vsh,
                                          box_center[2] + box_creation_table[i][2] * vsh);
                        this->m_queryPoints.push_back(QueryPoint<BaseVecT>(position, distances[i]));
                        box->setVertex(i, this->m_globalIndex);
                        this->m_globalIndex++;
                    }
                }
                // Set pointers to the neighbors of the current box
                int neighbor_index = 0;
                size_t neighbor_hash = 0;

                for (int a = -1; a < 2; a++)
                {
                    for (int b = -1; b < 2; b++)
                    {
                        for (int c = -1; c < 2; c++)
                        {

                            // Calculate hash value for current neighbor cell
                            neighbor_hash = this->hashValue(idx + a, idy + b, idz + c);

                            // Try to find this cell in the grid
                            auto neighbor_it = this->m_cells.find(neighbor_hash);

                            // If it exists, save pointer in box
                            if (neighbor_it != this->m_cells.end())
                            {
                                box->setNeighbor(neighbor_index, (*neighbor_it).second);
                                (*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
                            }

                            neighbor_index++;
                        }
                    }
                }

                this->m_cells[hash] = box;
            }
        }
        fclose(pFile);
    }
}

template<typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::HashGrid(std::vector<PointBufferPtr> chunks,
                                   std::vector<BoundingBox<BaseVecT>> innerBoxes,
                                   BoundingBox<BaseVecT>& boundingBox,
                                   float voxelSize)
        : m_boundingBox(boundingBox), m_globalIndex(0)
{
    unsigned int INVALID = BoxT::INVALID_INDEX;
    m_voxelsize = voxelSize;
    calcIndices();
    size_t numCells;
    boost::shared_array<float> centers;
    boost::shared_array<float> queryPoints;
    boost::shared_array<int> extruded;
    float vsh = 0.5 * this->m_voxelsize;
    size_t counter = 1;
    std::cout << timestamp.getElapsedTime() << "Number of Chunks: "<< chunks.size()<< std::endl;
    std::string comment = timestamp.getElapsedTime() + "Loading grid ";
    lvr2::ProgressBar progress(chunks.size(), comment);
    for (PointBufferPtr chunk : chunks)
    {
        unsigned int current_index = 0;
        boost::optional<unsigned int> optNumCells =  chunk->getAtomic<unsigned int>("num_voxel");
        boost::optional<Channel<int>> optExtruded = chunk->getChannel<int>("extruded");
        boost::optional<Channel<float>> optTSDF = chunk->getFloatChannel("tsdf_values");
        centers = chunk->getPointArray();

        if(optNumCells && optExtruded && optTSDF)
        {
            queryPoints = optTSDF.get().dataPtr();
            extruded = optExtruded.get().dataPtr();
            numCells = optNumCells.get();

            // get the min and max vector of the inner chunk bounding box
            BaseVecT innerChunkMin = innerBoxes.at(counter - 1).getMin();
            BaseVecT innerChunkMax = innerBoxes.at(counter - 1).getMax();

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

                size_t idx = calcIndex((centers[cellCount * 3 + 0] - m_boundingBox.getMin()[0]) / m_voxelsize);
                size_t idy = calcIndex((centers[cellCount * 3 + 1] - m_boundingBox.getMin()[1]) / m_voxelsize);
                size_t idz = calcIndex((centers[cellCount * 3 + 2] - m_boundingBox.getMin()[2]) / m_voxelsize);
                size_t hash = hashValue(idx, idy, idz);
                auto cell_it = this->m_cells.find(hash);
                if (cell_it == this->m_cells.end() && !extruded.get()[cellCount])
                {
                    BoxT* box = new BoxT(BaseVecT(centers[cellCount * 3 + 0], centers[cellCount * 3 + 1], centers[cellCount * 3 + 2]));
                    for (int i = 0; i < 8; i++)
                    {
                        current_index = this->findQueryPoint(i, idx, idy, idz);
                        if (current_index != INVALID)
                            box->setVertex(i, current_index);
                        else
                        {
                            BaseVecT position(centers[cellCount * 3 + 0] + box_creation_table[i][0] * vsh,
                                              centers[cellCount * 3 + 1] + box_creation_table[i][1] * vsh,
                                              centers[cellCount * 3 + 2] + box_creation_table[i][2] * vsh);
                            this->m_queryPoints.push_back(QueryPoint<BaseVecT>(position, queryPoints[cellCount * 8 + i]));
                            box->setVertex(i, this->m_globalIndex);
                            this->m_globalIndex++;
                        }
                    }
                    // Set pointers to the neighbors of the current box
                    int neighbor_index = 0;
                    size_t neighbor_hash = 0;

                    for (int a = -1; a < 2; a++)
                    {
                        for (int b = -1; b < 2; b++)
                        {
                            for (int c = -1; c < 2; c++)
                            {

                                // Calculate hash value for current neighbor cell
                                neighbor_hash = this->hashValue(idx + a, idy + b, idz + c);

                                // Try to find this cell in the grid
                                auto neighbor_it = this->m_cells.find(neighbor_hash);

                                // If it exists, save pointer in box
                                if (neighbor_it != this->m_cells.end())
                                {
                                    box->setNeighbor(neighbor_index, (*neighbor_it).second);
                                    (*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
                                }

                                neighbor_index++;
                            }
                        }
                    }

                    this->m_cells[hash] = box;
                }
            }
        }
        else
        {
            std::cout << "WARNING: something went wrong while reconstructing multiple chunks. Please check if all channels are available." << std::endl;
        }
        ++counter;
        if(!timestamp.isQuiet())
            ++progress;
    }
    if(!timestamp.isQuiet())
        cout << endl;
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::addLatticePoint(int index_x,
                                               int index_y,
                                               int index_z,
                                               float distance)
{
    size_t hash_value;

    unsigned int INVALID = BoxT::INVALID_INDEX;

    float vsh = 0.5 * this->m_voxelsize;

    // Some iterators for hash map accesses
    typename HashGrid<BaseVecT, BoxT>::box_map_it it;
    typename HashGrid<BaseVecT, BoxT>::box_map_it neighbor_it;

    // Values for current and global indices. Current refers to a
    // already present query point, global index is id that the next
    // created query point will get
    unsigned int current_index = 0;

    // Get min and max vertex of the point clouds bounding box
    auto v_min = this->m_boundingBox.getMin();
    auto v_max = this->m_boundingBox.getMax();

    int limit = this->m_extrude ? 1 : 0;
    for (int dx = -limit; dx <= limit; dx++)
    {
        for (int dy = -limit; dy <= limit; dy++)
        {
            for (int dz = -limit; dz <= limit; dz++)
            {
                hash_value = this->hashValue(index_x + dx, index_y + dy, index_z + dz);

                it = this->m_cells.find(hash_value);
                if (it == this->m_cells.end())
                {
                    // Calculate box center
                    BaseVecT box_center((index_x + dx) * this->m_voxelsize + v_min.x,
                                        (index_y + dy) * this->m_voxelsize + v_min.y,
                                        (index_z + dz) * this->m_voxelsize + v_min.z);

                    // WHY? this makes the results more worse than lvr1
                    // if((
                    //         box_center[0] <= m_boundingBox.getMin().x ||
                    //         box_center[1] <= m_boundingBox.getMin().y ||
                    //         box_center[2] <= m_boundingBox.getMin().z ||
                    //         box_center[0] >= m_boundingBox.getMax().x + m_voxelsize ||
                    //         box_center[1] >= m_boundingBox.getMax().y + m_voxelsize ||
                    //         box_center[2] >= m_boundingBox.getMax().z + m_voxelsize
                    // ))
                    // {
                    //     continue;
                    // }

                    // Create new box
                    BoxT* box = new BoxT(box_center);

                    if (box_center[0] <= m_boundingBox.getMin().x + m_voxelsize * 5 ||
                        box_center[1] <= m_boundingBox.getMin().y + m_voxelsize * 5 ||
                        box_center[2] <= m_boundingBox.getMin().z + m_voxelsize * 5)
                    {
                        box->m_duplicate = true;
                    }
                    else if (box_center[0] >= m_boundingBox.getMax().x - m_voxelsize * 5 ||
                             box_center[1] >= m_boundingBox.getMax().y - m_voxelsize * 5 ||
                             box_center[2] >= m_boundingBox.getMax().z - m_voxelsize * 5)
                    {
                        box->m_duplicate = true;
                    }

                    // Setup the box itself
                    for (int k = 0; k < 8; k++)
                    {

                        // Find point in Grid
                        current_index =
                            this->findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);
                        // If point exist, save index in box
                        if (current_index != INVALID)
                            box->setVertex(k, current_index);

                        // Otherwise create new grid point and associate it with the current box
                        else
                        {
                            BaseVecT position(box_center.x + box_creation_table[k][0] * vsh,
                                              box_center.y + box_creation_table[k][1] * vsh,
                                              box_center.z + box_creation_table[k][2] * vsh);

                            qp_bb.expand(position);

                            this->m_queryPoints.push_back(QueryPoint<BaseVecT>(position, distance));
                            box->setVertex(k, this->m_globalIndex);
                            this->m_globalIndex++;
                        }
                    }

                    // Set pointers to the neighbors of the current box
                    int neighbor_index = 0;
                    size_t neighbor_hash = 0;

                    for (int a = -1; a < 2; a++)
                    {
                        for (int b = -1; b < 2; b++)
                        {
                            for (int c = -1; c < 2; c++)
                            {

                                // Calculate hash value for current neighbor cell
                                neighbor_hash = this->hashValue(
                                    index_x + dx + a, index_y + dy + b, index_z + dz + c);

                                // Try to find this cell in the grid
                                neighbor_it = this->m_cells.find(neighbor_hash);

                                // If it exists, save pointer in box
                                if (neighbor_it != this->m_cells.end())
                                {
                                    box->setNeighbor(neighbor_index, (*neighbor_it).second);
                                    (*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
                                }

                                neighbor_index++;
                            }
                        }
                    }

                    this->m_cells[hash_value] = box;
                }
            }
        }
    }
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::setCoordinateScaling(float x, float y, float z)
{
    m_coordinateScales.x = x;
    m_coordinateScales.y = y;
    m_coordinateScales.z = z;
}

template <typename BaseVecT, typename BoxT>
HashGrid<BaseVecT, BoxT>::~HashGrid()
{
    box_map_it iter;
    for (iter = m_cells.begin(); iter != m_cells.end(); iter++)
    {
        if (iter->second != NULL)
        {
            delete (iter->second);
            iter->second = NULL;
        }
    }

    m_cells.clear();
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::calcIndices()
{
    float max_size = m_boundingBox.getLongestSide();

    // Save needed grid parameters
    m_maxIndex = (int)ceil((max_size + 5 * m_voxelsize) / m_voxelsize);
    m_maxIndexSquare = m_maxIndex * m_maxIndex;

    m_maxIndexX = (int)ceil(m_boundingBox.getXSize() / m_voxelsize) + 1;
    m_maxIndexY = (int)ceil(m_boundingBox.getYSize() / m_voxelsize) + 2;
    m_maxIndexZ = (int)ceil(m_boundingBox.getZSize() / m_voxelsize) + 3;
}

template <typename BaseVecT, typename BoxT>
unsigned int HashGrid<BaseVecT, BoxT>::findQueryPoint(int position, int x, int y, int z)
{
    int n_x, n_y, n_z, q_v, offset;
    box_map_it it;

    for (int i = 0; i < 7; i++)
    {
        offset = i * 4;
        n_x = x + shared_vertex_table[position][offset];
        n_y = y + shared_vertex_table[position][offset + 1];
        n_z = z + shared_vertex_table[position][offset + 2];
        q_v = shared_vertex_table[position][offset + 3];

        size_t hash = hashValue(n_x, n_y, n_z);
        // cout << "i=" << i << " looking for hash: " << hash << endl;
        it = m_cells.find(hash);
        if (it != m_cells.end())
        {
            //  cout << "found hash" << endl;
            BoxT* b = it->second;
            if (b->getVertex(q_v) != BoxT::INVALID_INDEX)
                return b->getVertex(q_v);
        }
        // cout << "did not find hash" << endl;
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
        typename unordered_map<size_t, BoxT*>::iterator it;
        BoxT* box;
        for (it = m_cells.begin(); it != m_cells.end(); it++)
        {
            box = it->second;
            for (int i = 0; i < 8; i++)
            {
                out << box->getVertex(i) << " ";
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
    for (auto it = this->firstCell(); it != this->lastCell(); it++)
    {
        fwrite(&it->second->getCenter()[0], sizeof(float), 1, pFile);
        fwrite(&it->second->getCenter()[1], sizeof(float), 1, pFile);
        fwrite(&it->second->getCenter()[2], sizeof(float), 1, pFile);

        fwrite(&it->second->m_extruded, sizeof(bool), 1, pFile);

        fwrite(&m_queryPoints[it->second->getVertex(0)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[it->second->getVertex(1)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[it->second->getVertex(2)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[it->second->getVertex(3)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[it->second->getVertex(4)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[it->second->getVertex(5)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[it->second->getVertex(6)].m_distance, sizeof(float), 1, pFile);
        fwrite(&m_queryPoints[it->second->getVertex(7)].m_distance, sizeof(float), 1, pFile);
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
        typename unordered_map<size_t, BoxT*>::iterator it;
        BoxT* box;
        for (it = m_cells.begin(); it != m_cells.end(); it++)
        {
            box = it->second;
            out << it->first << " ";
            for (int i = 0; i < 8; i++)
            {
                out << box->getVertex(i) << " ";
            }
            out << box->getCenter().x << " " << box->getCenter().y << " " << box->getCenter().z
                << " " << box->m_extruded << endl;
        }
    }
    out.close();
    std::cout << timestamp << "finished saving grid: " << file << std::endl;
}

template <typename BaseVecT, typename BoxT>
void HashGrid<BaseVecT, BoxT>::setBB(BoundingBox<BaseVecT>& bb)
{
    m_boundingBox = bb;
    calcIndices();
}

} // namespace lvr2
