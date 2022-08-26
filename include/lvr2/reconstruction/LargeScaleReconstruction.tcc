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

#include <iostream>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/geometry/PMPMesh.hpp"
#include "lvr2/algorithm/pmp/SurfaceNormals.h"
#include "lvr2/io/baseio/ChannelIO.hpp"
#include "lvr2/io/baseio/ArrayIO.hpp"
#include "lvr2/io/baseio/VariantChannelIO.hpp"
#include "lvr2/io/Tiles3dIO.hpp"
#include "lvr2/reconstruction/VirtualGrid.hpp"
#include "lvr2/reconstruction/BigGridKdTree.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/registration/OctreeReduction.hpp"

#include "lvr2/algorithm/CleanupAlgorithms.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/algorithm/Tesselator.hpp"
#include "lvr2/util/Timestamp.hpp"


#include "LargeScaleReconstruction.hpp"

#include <mpi.h>
#include <yaml-cpp/yaml.h>


#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#endif

namespace lvr2
{
    using LSRWriter = lvr2::Hdf5IO<lvr2::hdf5features::ArrayIO,
            lvr2::hdf5features::ChannelIO,
            lvr2::hdf5features::VariantChannelIO,
            lvr2::hdf5features::MeshIO>;

    template<typename BaseVecT>
    LargeScaleReconstruction<BaseVecT>::LargeScaleReconstruction(LSROptions options)
        : m_options(options)
    {
        m_options.ensureCorrectness();
        std::cout << timestamp << "Reconstruction Instance generated..." << std::endl;
    }

    template <typename BaseVecT>
    void LargeScaleReconstruction<BaseVecT>::chunkAndReconstructVGrid(
        ScanProjectEditMarkPtr project,
        BoundingBox<BaseVecT>& newChunksBB,
        std::shared_ptr<ChunkHashGrid> chunkManager)
    {
        auto startTimeMs = timestamp.getCurrentTimeInMs();

        if(project->project->positions.size() != project->changed.size())
        {
            throw std::runtime_error("Inconsistency between number of given scans and diff-vector (scans to consider)!");
        }

        /// Minimum number of points needed to consider a chunk. Chunks smaller than this are skipped.
        size_t minPointsPerChunk = std::max(m_options.ki, std::max(m_options.kd, m_options.kn)) * 2;

        /// Maximum number of points in a chunk to avoid GPU memory overflow. Chunks bigger than this are reduced.
        size_t maxPointsPerChunk = 130'000'000;

        float chunkSize = m_options.bgVoxelSize;

        std::cout << timestamp << "Starting BigGrid" << std::endl;
        BigGrid<BaseVecT> bg(chunkSize, project, m_options.tempDir, m_options.scale);
        std::cout << timestamp << "BigGrid finished " << std::endl;

        BoundingBox<BaseVecT> bgBB = bg.getBB();

        std::vector<BoundingBox<BaseVecT>> partitionBoxes;
        std::vector<BoundingBox<BaseVecT>> filteredPartitionBoxes;
        std::vector<Vector3i> chunkCoords;

        // BigGrid is essentially a V-Grid that already knows where all the points are,
        // so just recycle the cells as partitions.
        auto& cells = bg.getCells();
        partitionBoxes.reserve(cells.size());

        // ChunkManager BoundingBox: must contain all old chunks and newly added chunks
        BoundingBox<BaseVecT> cmBB = bgBB;
        // newChunks BoundingBox: contains only newly added chunks
        newChunksBB = BoundingBox<BaseVecT>();

        BaseVecT cellSize(chunkSize, chunkSize, chunkSize);
        for (auto& [ index, cell ] : cells)
        {
            BaseVecT pos = BaseVecT(index.x(), index.y(), index.z()) * chunkSize;
            BoundingBox<BaseVecT> bb(pos, pos + cellSize);
            newChunksBB.expand(bb);
            partitionBoxes.push_back(bb);
            chunkCoords.push_back(index);
        }
        cmBB.expand(newChunksBB);
        chunkManager->setBoundingBox(cmBB);

        std::cout << timestamp << "got: " << bg.getCells().size() << " chunks." << std::endl;

        std::cout << lvr2::timestamp << "VoxelSizes: ";
        for (auto v : m_options.voxelSizes)
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;

        for(size_t h = 0; h < m_options.voxelSizes.size(); h++)
        {
            float voxelSize = m_options.voxelSizes[h];
            float overlap = 5 * voxelSize;
            BaseVecT overlapVector(overlap, overlap, overlap);

            bool createBigMesh = false, createChunksPly = false, createChunksHdf5 = false, create3dTiles = false;
            // only produce output for the first voxel size
            if (h == 0)
            {
                createBigMesh = m_options.hasOutput(LSROutput::BigMesh);
                createChunksPly = m_options.hasOutput(LSROutput::ChunksPly);
                createChunksHdf5 = m_options.hasOutput(LSROutput::ChunksHdf5);

#ifdef LVR2_USE_3DTILES
                create3dTiles = m_options.hasOutput(LSROutput::Tiles3d);
#endif
            }

            // file to store chunks in for LSROutput::ChunksHdf5
            std::shared_ptr<HighFive::File> chunkFileHdf5 = nullptr;
            if (createChunksHdf5)
            {
                chunkFileHdf5 = hdf5util::open(m_options.outputDir / "chunks.h5", HighFive::File::Truncate);
                auto root = chunkFileHdf5->getGroup("/");

                hdf5util::setAttribute(root, "chunk_size", chunkSize);
                hdf5util::setAttribute(root, "voxel_size", voxelSize);
            }

            // directory to store chunks in for LSROutput::ChunksPly
            fs::path chunkDirPly = "";
            if (createChunksPly)
            {
                chunkDirPly = m_options.outputDir / "chunks/";
                fs::remove_all(chunkDirPly);
                fs::create_directories(chunkDirPly);

                YAML::Node metadata;
                metadata["chunk_size"] = chunkSize;
                metadata["voxel_size"] = voxelSize;

                std::ofstream out(chunkDirPly / "chunk_metadata.yaml");
                out << metadata;
            }

            // collection of all chunks for LSROutput::Tiles3d
            typename HLODTree<BaseVecT>::ChunkMap chunkMap;
            // file to store chunks in for LSROutput::Tiles3d
            std::shared_ptr<HighFive::File> chunkFile3dTiles = nullptr;
            if (create3dTiles && m_options.tiles3dMemUsage < AllowedMemoryUsage::Unbounded)
            {
                chunkFile3dTiles = hdf5util::open(m_options.tempDir / "lazy_meshes.h5", HighFive::File::Truncate);
            }

            // List of chunks that should actually be reconstructed
            std::vector<Vector3i> filteredChunkCoords;

            string layerName = "tsdf_values_" + std::to_string(voxelSize);
            string layerNameTemp = layerName + "_temp";

            for (size_t i = 0; i < partitionBoxes.size(); i++)
            {
                auto& partitionBox = partitionBoxes[i];
                auto& coord = chunkCoords[i];

                std::cout << timestamp << "Processing Partition " << i << "/" << (partitionBoxes.size() - 1) << std::endl;

                BoundingBox<BaseVecT> gridbb(partitionBox.getMin() - overlapVector, partitionBox.getMax() + overlapVector);

                bool retry = false;
                GridPtr ps_grid;
                do
                {
                    ps_grid = createChunk(bg, gridbb, voxelSize, minPointsPerChunk, maxPointsPerChunk, retry);
                } while (retry);

                if (!ps_grid)
                {
                    continue;
                }

                // all checks are done, this partition is ok
                filteredPartitionBoxes.push_back(partitionBox);
                filteredChunkCoords.push_back(coord);

                if (m_options.mergeChunkBorders)
                {
                    chunkManager->setChunk(layerNameTemp, coord.x(), coord.y(), coord.z(), ps_grid->toPointBuffer());
                }
                else
                {
                    if (createBigMesh)
                    {
                        chunkManager->setChunk(layerName, coord.x(), coord.y(), coord.z(), ps_grid->toPointBuffer());
                    }
                    if (createChunksHdf5 || createChunksPly || create3dTiles)
                    {
                        processChunk(ps_grid, coord, chunkDirPly, chunkFileHdf5, chunkFile3dTiles, chunkMap);
                    }
                }
            }

            if (m_options.mergeChunkBorders)
            {
                std::cout << timestamp << "Finished calculating TSDFs. Merging chunk overlaps" << std::endl;

                // an empty grid to call calcIndex on
                HashGrid<BaseVecT, BoxT> dummyGrid(voxelSize, {}, true, m_options.extrude);

                for (size_t i = 0; i < filteredPartitionBoxes.size(); i++)
                {
                    auto& partitionBox = filteredPartitionBoxes[i];
                    auto& coord = filteredChunkCoords[i];

                    std::cout << timestamp << "Processing Partition " << i << "/" << (filteredPartitionBoxes.size() - 1) << std::endl;

                    // In order to get consistent chunks, it is necessary to have the exact same
                    // distance values on both sides of a border.
                    // To make sure this is the case, we merge the values of the overlaps for all
                    // neighboring chunks.
                    GridPtr ps_grid;
                    std::vector<std::unordered_map<Vector3i, float>> neighborQPs;
                    std::vector<BoundingBox<BaseVecT>> neighborBBs;
                    // QueryPoints are on the corners of the cells, and rounding errors can cause
                    // them to land in either cell when using calcIndex. So we shift them by half a voxel.
                    BaseVecT halfVoxel(voxelSize / 2, voxelSize / 2, voxelSize / 2);

                    #pragma omp parallel for collapse(3)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                boost::optional<PointBufferPtr> optPointBuffer;
                                #pragma omp critical
                                optPointBuffer = chunkManager->getChunk<PointBufferPtr>(layerNameTemp, coord.x() + dx, coord.y() + dy, coord.z() + dz);
                                if (!optPointBuffer)
                                {
                                    continue;
                                }
                                auto pointBuffer = *optPointBuffer;
                                BaseVecT pos = BaseVecT(coord.x() + dx, coord.y() + dy, coord.z() + dz) * chunkSize;
                                BoundingBox<BaseVecT> bb(pos - overlapVector, pos + cellSize + overlapVector);
                                if (dx == 0 && dy == 0 && dz == 0)
                                {
                                    ps_grid = std::make_shared<HashGrid<BaseVecT, BoxT>>(pointBuffer, bb, voxelSize);
                                    continue;
                                }

                                size_t numCells = pointBuffer->numPoints();
                                auto centers = *pointBuffer->getFloatChannel("points");
                                float* distances = pointBuffer->getFloatChannel("tsdf_values")->dataPtr().get();

                                std::unordered_map<Vector3i, float> qps;
                                Vector3i index;
                                for (size_t i = 0; i < numCells; i++)
                                {
                                    float* dist = distances + 8 * i;
                                    BaseVecT center = centers[i];
                                    for (int j = 0; j < 8; j++)
                                    {
                                        const int* table = box_creation_table[j];
                                        BaseVecT corner = center + BaseVecT(table[0], table[1], table[2]) * (voxelSize / 2);
                                        dummyGrid.calcIndex(corner + halfVoxel, index);
                                        qps[index] = dist[j];
                                    }
                                }

                                #pragma omp critical
                                {
                                    neighborQPs.emplace_back(std::move(qps));
                                    neighborBBs.emplace_back(bb);
                                }
                            }
                        }
                    }

                    if (!neighborQPs.empty())
                    {
                        std::cout << timestamp << "Merging " << neighborQPs.size() << " neighbors" << std::endl;

                        // QueryPoints are on the corners of the cells, and rounding errors can cause
                        // them to land in either cell when using calcIndex. So we shift them by half a voxel.
                        BaseVecT halfVoxel(voxelSize / 2, voxelSize / 2, voxelSize / 2);
                        auto& queryPoints = ps_grid->getQueryPoints();
                        #pragma omp parallel
                        {
                            // list of <center, distance> pairs for each neighbor
                            std::vector<std::pair<BaseVecT, float>> identical;
                            Vector3i index;
                            #pragma omp for schedule(dynamic,64)
                            for (size_t i = 0; i < queryPoints.size(); i++)
                            {
                                auto& pos = queryPoints[i].m_position;
                                ps_grid->calcIndex(pos + halfVoxel, index);
                                identical.clear();
                                for (size_t j = 0; j < neighborQPs.size(); j++)
                                {
                                    auto it = neighborQPs[j].find(index);
                                    if (it != neighborQPs[j].end())
                                    {
                                        auto center = neighborBBs[j].getCentroid();
                                        identical.emplace_back(center, it->second);
                                    }
                                }
                                if (identical.empty())
                                {
                                    continue;
                                }
                                // also add the query point itself
                                auto center = ps_grid->getBoundingBox().getCentroid();
                                identical.emplace_back(center, queryPoints[i].m_distance);

                                double totalDistance = 0;
                                double totalWeight = 0;
                                for (auto& [ center, distance ] : identical)
                                {
                                    // weighting function:
                                    // left neighbor    |    current cell     |    right neighbor
                                    //             | overlap |           | overlap |
                                    // weight:      0   ->   1 ... 1 ... 1   <-   0
                                    double weight = 1.0;
                                    for (int axis = 0; axis < 3; axis++)
                                    {
                                        double distFromEdge = chunkSize / 2.0 + overlap - std::abs(center[axis] - pos[axis]);
                                        double axisWeight = distFromEdge / (2 * overlap);
                                        weight = std::min(weight, axisWeight);
                                    }
                                    weight = std::max(weight, 0.001); // avoid division by zero
                                    totalDistance += distance * weight;
                                    totalWeight += weight;
                                }
                                queryPoints[i].m_distance = totalDistance / totalWeight;
                            }
                        }
                    }

                    if (createBigMesh)
                    {
                        chunkManager->setChunk(layerName, coord.x(), coord.y(), coord.z(), ps_grid->toPointBuffer());
                    }
                    if (createChunksHdf5 || createChunksPly || create3dTiles)
                    {
                        processChunk(ps_grid, coord, chunkDirPly, chunkFileHdf5, chunkFile3dTiles, chunkMap);
                    }
                }
            }
            size_t skipped = partitionBoxes.size() - filteredPartitionBoxes.size();
            std::cout << timestamp << "Skipped PartitionBoxes: " << skipped << std::endl;

            std::cout << timestamp << "finished" << std::endl;

            if (createChunksHdf5)
            {
                // make sure Chunks are properly saved, in case the following code crashes
                chunkFileHdf5->flush();
                H5Fclose(chunkFileHdf5->getId());
                chunkFileHdf5.reset();
            }

#ifdef LVR2_USE_3DTILES
            if (create3dTiles && !chunkMap.empty())
            {
                std::cout << timestamp << "Creating 3D Tiles: Generating HLOD Tree" << std::endl;
                if (m_options.tiles3dMemUsage > AllowedMemoryUsage::Minimal)
                {
                    std::cout << timestamp << "Note: If the following overflows your RAM, try setting 3dTilesMemUsage to Minimal" << std::endl;
                }
                auto tree = HLODTree<BaseVecT>::partition(std::move(chunkMap), 3);
                tree->finalize(0.2, m_options.tiles3dMemUsage);

                std::cout << timestamp << "Creating 3D Tiles: Writing to mesh.3dtiles" << std::endl;
                Tiles3dIO<BaseVecT> io((m_options.outputDir / "mesh.3dtiles").string());
                io.write(tree, m_options.tiles3dCompress);
                tree.reset();

                std::cout << timestamp << "Creating 3D Tiles: Finished" << std::endl;
            }
            if (chunkFile3dTiles)
            {
                std::string filename = chunkFile3dTiles->getName();
                chunkFile3dTiles.reset();
                boost::filesystem::remove(filename);
            }
#endif // LVR2_USE_3DTILES

            if(createBigMesh)
            {
                //combine chunks
                BoundingBox<BaseVecT> cbb(bgBB.getMin() - overlapVector, bgBB.getMax() + overlapVector);

                std::vector<PointBufferPtr> tsdfChunks;
                for (auto& coord : filteredChunkCoords)
                {
                    auto chunk = chunkManager->getChunk<PointBufferPtr>(layerName, coord.x(), coord.y(), coord.z());
                    if (chunk)
                    {
                        tsdfChunks.push_back(chunk.get());
                    }
                    else
                    {
                        std::cout << "WARNING - Could not find chunk (" << coord.x() << ", " << coord.y() << ", " << coord.z()
                                << ") in layer: " << "tsdf_values_" + std::to_string(voxelSize) << std::endl;
                    }
                }
                auto hg = std::make_shared<HashGrid<BaseVecT, BoxT>>(tsdfChunks, filteredPartitionBoxes, cbb, voxelSize);
                tsdfChunks.clear();

                createAndSaveBigMesh(hg, h);
            }

            std::cout << lvr2::timestamp << "added/changed " << filteredChunkCoords.size() << " chunks in layer " << layerName << std::endl;
        }

        auto timeDiffMs = lvr2::timestamp.getCurrentTimeInMs() - startTimeMs;

        cout << "Finished complete reconstruction in " << (timeDiffMs / 1000.0) << "s" << endl;
    }

    template <typename BaseVecT>
    void LargeScaleReconstruction<BaseVecT>::chunkAndReconstructKDTree(
        ScanProjectEditMarkPtr project,
        BoundingBox<BaseVecT>& newChunksBB)
    {
        auto startTimeMs = timestamp.getCurrentTimeInMs();

        if(project->project->positions.size() != project->changed.size())
        {
            throw std::runtime_error("Inconsistency between number of given scans and diff-vector (scans to consider)!");
        }

        pmp::Point flipPoint(m_options.flipPoint[0], m_options.flipPoint[1], m_options.flipPoint[2]);

        /// Minimum number of points needed to consider a chunk. Chunks smaller than this are skipped.
        size_t minPointsPerChunk = std::max(m_options.ki, std::max(m_options.kd, m_options.kn)) * 2;

        /// Maximum number of points in a chunk to avoid GPU memory overflow. Chunks bigger than this are reduced.
        size_t maxPointsPerChunk = 130'000'000;

        std::cout << timestamp << "Starting BigGrid" << std::endl;
        BigGrid<BaseVecT> bg(m_options.bgVoxelSize, project, m_options.tempDir, m_options.scale);
        std::cout << timestamp << "BigGrid finished " << std::endl;

        BoundingBox<BaseVecT> bgBB = bg.getBB();

        std::vector<BoundingBox<BaseVecT>> partitionBoxes;
        std::vector<BoundingBox<BaseVecT>> filteredPartitionBoxes;

        // use KD-Tree
        std::cout << timestamp << "generating tree" << std::endl;
        BigGridKdTree<BaseVecT> gridKd(bg.getBB(), m_options.nodeSize, &bg, m_options.bgVoxelSize);
        gridKd.insert(bg.pointSize(), bg.getBB().getCentroid());
        auto leafs = gridKd.getLeafs();
        partitionBoxes.reserve(leafs.size());
        for (auto& leaf : leafs)
        {
            partitionBoxes.push_back(leaf->getBB());
        }

        std::cout << timestamp << "Finished tree" << std::endl;

        std::cout << timestamp << "got: " << bg.getCells().size() << " chunks." << std::endl;

        std::cout << lvr2::timestamp << "VoxelSizes: ";
        for (auto v : m_options.voxelSizes)
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;

        for(size_t h = 0; h < m_options.voxelSizes.size(); h++)
        {
            float voxelSize = m_options.voxelSizes[h];
            float overlap = 5 * voxelSize;
            BaseVecT overlapVector(overlap, overlap, overlap);

            // Vector to store relevant chunks as .ser
            std::vector<string> grid_files;

            for (size_t i = 0; i < partitionBoxes.size(); i++)
            {
                auto& partitionBox = partitionBoxes[i];
                string name_id = std::to_string(i);

                BoundingBox<BaseVecT> gridbb(partitionBox.getMin() - overlapVector, partitionBox.getMax() + overlapVector);

                bool retry = false;
                GridPtr ps_grid;
                do
                {
                    ps_grid = createChunk(bg, gridbb, voxelSize, minPointsPerChunk, maxPointsPerChunk, retry);
                } while (retry);

                if (!ps_grid)
                {
                    continue;
                }

                // all checks are done, this partition is ok
                filteredPartitionBoxes.push_back(partitionBox);

                std::string filename = (m_options.tempDir / (name_id + ".ser")).string();
                ps_grid->saveGrid(filename);
                grid_files.push_back(filename);
            }
            size_t skipped = partitionBoxes.size() - filteredPartitionBoxes.size();
            std::cout << timestamp << "Skipped PartitionBoxes: " << skipped << std::endl;

            std::cout << timestamp << "finished" << std::endl;

            //combine chunks
            BoundingBox<BaseVecT> cbb(bgBB.getMin() - overlapVector, bgBB.getMax() + overlapVector);

            auto hg = std::make_shared<HashGrid<BaseVecT, BoxT>>(grid_files, filteredPartitionBoxes, cbb, voxelSize);
            createAndSaveBigMesh(hg, h);
        }

        auto timeDiffMs = lvr2::timestamp.getCurrentTimeInMs() - startTimeMs;

        cout << "Finished complete reconstruction in " << (timeDiffMs / 1000.0) << "s" << endl;
    }

    template<typename BaseVecT>
    typename LargeScaleReconstruction<BaseVecT>::GridPtr LargeScaleReconstruction<BaseVecT>::createChunk(
        const BigGrid<BaseVecT>& bg,
        const BoundingBox<BaseVecT>& bb,
        float voxelSize,
        size_t minPointsPerChunk,
        size_t& maxPointsPerChunk,
        bool& retry)
    {
        retry = false;
        if (bg.estimateSizeofBox(bb) < minPointsPerChunk)
        {
            return nullptr;
        }

        size_t numPoints;
        floatArr points = bg.points(bb, numPoints, minPointsPerChunk);

        if (!points)
        {
            return nullptr;
        }

        auto p_loader = std::make_shared<PointBuffer>(points, numPoints);

        bool hasNormals = false;
        bool hasDistances = false;

        if (!hasNormals && bg.hasNormals())
        {
            size_t numNormals;
            lvr2::floatArr normals = bg.normals(bb, numNormals);
            if (numNormals != numPoints)
            {
                throw std::runtime_error("Number of normals does not match number of points");
            }
            p_loader->setNormalArray(normals, numNormals);
            hasNormals = true;
        }

        if (!hasNormals && m_options.useGPU)
        {
            float targetSize = voxelSize / 4;
            while (numPoints > maxPointsPerChunk)
            {
                // reduction is necessary to avoid GPU memory overflow
                std::cout << timestamp << "Chunk has too many points: " << numPoints << ". Reducing." << std::endl;
                OctreeReduction oct(p_loader, targetSize, 10);
                p_loader = oct.getReducedPoints();
                numPoints = p_loader->numPoints();
                targetSize *= 2; // bigger voxel size means more points removed during reduction in the next iteration
            }
        }

        BaseVecT flipPoint(m_options.flipPoint[0], m_options.flipPoint[1], m_options.flipPoint[2]);

        lvr2::PointsetSurfacePtr<BaseVecT> surface;
        {
            auto ptr = new AdaptiveKSearchSurface<BaseVecT>(p_loader, "lvr2", m_options.kn, m_options.ki, m_options.kd, m_options.useRansac);
            ptr->setFlipPoint(flipPoint);
            surface.reset(ptr);
        }

        auto ps_grid = std::make_shared<lvr2::PointsetGrid<BaseVecT, BoxT>>(voxelSize, surface, bb, true, m_options.extrude);


#ifdef GPU_FOUND
        if ((!hasNormals && m_options.useGPU) || (!hasDistances && m_options.useGPUDistances))
        {
            floatArr points = p_loader->getPointArray();
            std::cout << timestamp << "Generate GPU kd-tree..." << std::endl;

            GpuSurface gpu_surface(points, numPoints);

            gpu_surface.setKn(m_options.kn);
            gpu_surface.setKi(m_options.ki);
            gpu_surface.setKd(m_options.kd);
            gpu_surface.setFlippoint(flipPoint.x, flipPoint.y, flipPoint.z);

            if (!hasNormals && m_options.useGPU)
            {
                floatArr normals = floatArr(new float[numPoints * 3]);
                try
                {
                    gpu_surface.calculateNormals();
                }
                catch(std::runtime_error& e)
                {
                    std::string msg = e.what();
                    if (msg.find("out of memory") == std::string::npos)
                    {
                        throw; // forward any other exceptions
                    }
                    std::cerr << "ERROR: Not enough GPU memory. Reducing Points further." << std::endl;
                    maxPointsPerChunk = maxPointsPerChunk * 0.8;
                    if (maxPointsPerChunk < minPointsPerChunk)
                    {
                        std::cerr << "Your GPU is garbage. Switching back to CPU" << std::endl;
                        m_options.useGPU = false;
                    }
                    retry = true;
                    return nullptr;
                }
                gpu_surface.getNormals(normals);

                p_loader->setNormalArray(normals, numPoints);
                hasNormals = true;
            }

            if(!hasDistances && m_options.useGPUDistances)
            {
                auto& query_points = ps_grid->getQueryPoints();

                std::cout << timestamp << "Computing signed distances in GPU with brute force kernel." << std::endl;
                std::cout << timestamp << "This might take a while...." << std::endl;
                gpu_surface.distances(query_points, voxelSize);
                hasDistances = true;
                std::cout << timestamp << "Done." << std::endl;
            }
        }
#endif // GPU_FOUND

        if (!hasNormals)
        {
            surface->calculateSurfaceNormals();
        }
        if (!hasDistances)
        {
            ps_grid->calcDistanceValues();
        }

        if (ps_grid->getCells().empty())
        {
            return nullptr;
        }

        return ps_grid;
    }

    template<typename BaseVecT>
    void LargeScaleReconstruction<BaseVecT>::processChunk(
        GridPtr ps_grid,
        const Vector3i& coord,
        const fs::path& chunkDirPly,
        std::shared_ptr<HighFive::File> chunkFileHdf5,
        std::shared_ptr<HighFive::File> chunkFile3dTiles,
        typename HLODTree<BaseVecT>::ChunkMap& chunkMap)
    {
        lvr2::FastReconstruction<BaseVecT, BoxT> reconstruction(ps_grid);
        lvr2::PMPMesh<BaseVecT> mesh;
        reconstruction.getMesh(mesh);

        if (mesh.numFaces() == 0)
        {
            return;
        }

        // run cleanup on the mesh.
        // cleanContours and optimizePlanes might mess with the chunk borders

        if (m_options.removeDanglingArtifacts)
        {
            std::cout << timestamp << "Removing dangling artifacts" << std::endl;
            removeDanglingCluster(mesh, m_options.removeDanglingArtifacts);
            if (mesh.numFaces() == 0)
            {
                return;
            }
        }
        if (m_options.fillHoles)
        {
            mesh.fillHoles(m_options.fillHoles);
        }
        mesh.collectGarbage();

        std::string name_id = std::to_string(coord.x()) + "_" + std::to_string(coord.y()) + "_" + std::to_string(coord.z());

        auto& surfaceMesh = mesh.getSurfaceMesh();
        if (m_options.hasOutput(LSROutput::ChunksHdf5))
        {
            auto group = chunkFileHdf5->createGroup("/chunks/" + name_id);
            surfaceMesh.write(group);
            chunkFileHdf5->flush();
        }

        if (m_options.hasOutput(LSROutput::ChunksPly))
        {
            surfaceMesh.write((chunkDirPly / (name_id + ".ply")).string());
        }

        #ifdef LVR2_USE_3DTILES
        if (m_options.hasOutput(LSROutput::Tiles3d))
        {
            pmp::Point flipPoint(m_options.flipPoint[0], m_options.flipPoint[1], m_options.flipPoint[2]);
            pmp::SurfaceNormals::compute_vertex_normals(surfaceMesh, flipPoint);

            pmp::Point epsilon = pmp::Point::Constant(0.0001);
            pmp::Point min = coord.cast<float>() * m_options.bgVoxelSize;
            pmp::Point max = min + pmp::Point::Constant(m_options.bgVoxelSize);
            pmp::BoundingBox expectedBB(min - epsilon, max + epsilon);

            HLODTree<BaseVecT>::trimChunkOverlap(mesh, expectedBB);

            if (mesh.numFaces() > 0)
            {
                auto bb = surfaceMesh.bounds();
                chunkMap.emplace(coord, HLODTree<BaseVecT>::leaf(LazyMesh(std::move(mesh), chunkFile3dTiles), bb));
            }
        }
        #endif 
    }

    template<typename BaseVecT>
    void LargeScaleReconstruction<BaseVecT>::createAndSaveBigMesh(GridPtr hg, size_t voxelSizeIndex)
    {
        auto reconstruction = make_unique<lvr2::FastReconstruction<BaseVecT, BoxT>>(hg);

        lvr2::PMPMesh<BaseVecT> mesh;

        reconstruction->getMesh(mesh);
        reconstruction.reset();
        hg.reset();

        if (m_options.removeDanglingArtifacts)
        {
            cout << timestamp << "Removing dangling artifacts" << endl;
            removeDanglingCluster(mesh, m_options.removeDanglingArtifacts);
        }

        // Magic number from lvr1 `cleanContours`...
        cleanContours(mesh, m_options.cleanContours, 0.0001);

        // Fill small holes if requested
        if (m_options.fillHoles)
        {
            mesh.fillHoles(m_options.fillHoles);
        }

        if (m_options.optimizePlanes)
        {
            auto faceNormals = calcFaceNormals(mesh);
            auto clusterBiMap = iterativePlanarClusterGrowing(mesh, faceNormals, m_options.planeNormalThreshold, m_options.planeIterations, m_options.minPlaneSize);

            if (m_options.smallRegionThreshold > 0)
            {
                deleteSmallPlanarCluster(mesh, clusterBiMap, static_cast<size_t>(m_options.smallRegionThreshold));
            }

            if (m_options.retesselate)
            {
                Tesselator<BaseVecT>::apply(mesh, clusterBiMap, faceNormals, m_options.lineFusionThreshold);
            }
        }

        if (mesh.numFaces() > 0)
        {
            std::string suffix = voxelSizeIndex > 0 ? std::to_string(m_options.voxelSizes[voxelSizeIndex]) : "";
            fs::path filename = m_options.outputDir / ("mesh" + suffix + ".ply");
            std::cout << timestamp << "Writing mesh to " << filename << std::endl;
            mesh.getSurfaceMesh().write(filename.string());
        }
        else
        {
            std::cout << timestamp << "Warning: Mesh is empty!" << std::endl;
        }
    }

    template<typename BaseVecT>
    HalfEdgeMesh<BaseVecT> LargeScaleReconstruction<BaseVecT>::getPartialReconstruct(BoundingBox<BaseVecT> newChunksBB,
                                                                            std::shared_ptr<ChunkHashGrid> chunkHashGrid,
                                                                            float voxelSize) {
        string layerName = "tsdf_values_" + std::to_string(voxelSize);
        float chunkSize = chunkHashGrid->getChunkSize();

        std::vector<PointBufferPtr> tsdfChunks;
        std::vector<BoundingBox<BaseVecT>> filteredPartitionBoxes;
        BoundingBox<BaseVecT> completeBB = BoundingBox<BaseVecT>();

        int xMin = (int) (newChunksBB.getMin().x / chunkSize);
        int yMin = (int) (newChunksBB.getMin().y / chunkSize);
        int zMin = (int) (newChunksBB.getMin().z / chunkSize);

        int xMax = (int) (newChunksBB.getMax().x / chunkSize);
        int yMax = (int) (newChunksBB.getMax().y / chunkSize);
        int zMax = (int) (newChunksBB.getMax().z / chunkSize);

        std::cout << "DEBUG: New Chunks from (" << xMin << ", " << yMin << ", " << zMin
                  << ") - to (" << xMax << ", " << yMax << ", " << zMax << ")." << std::endl;

        for (int i = xMin - 1; i <= xMax + 1; i++) {
            for (int j = yMin - 1; j <= yMax + 1; j++) {
                for (int k = zMin - 1; k <= zMax + 1; k++) {
                    boost::optional<shared_ptr<PointBuffer>> chunk = chunkHashGrid->getChunk<PointBufferPtr>(layerName,
                                                                                                             i, j, k);


                    if (chunk) {
                        BaseVecT min(i * chunkSize, j * chunkSize, k * chunkSize);
                        BaseVecT max((i + 1) * chunkSize, (j + 1) * chunkSize, (k + 1) * chunkSize);

                        BoundingBox<BaseVecT> temp(min, max);
                        filteredPartitionBoxes.push_back(temp);
                        completeBB.expand(temp);

                        tsdfChunks.push_back(chunk.get());
                    }
                }
            }
        }


        auto hg = std::make_shared<HashGrid<BaseVecT, BoxT>>(tsdfChunks, filteredPartitionBoxes, completeBB,
                                                                           voxelSize);
        tsdfChunks.clear();
        auto reconstruction = make_unique<lvr2::FastReconstruction<BaseVecT, BoxT>>(hg);

        lvr2::HalfEdgeMesh<BaseVecT> mesh;
        reconstruction->getMesh(mesh);

        if (m_options.removeDanglingArtifacts) {
            cout << timestamp << "Removing dangling artifacts" << endl;
            removeDanglingCluster(mesh, static_cast<size_t>(m_options.removeDanglingArtifacts));
        }

        if (m_options.fillHoles) {
            naiveFillSmallHoles(mesh, m_options.fillHoles, false);
        }


        if (m_options.hasOutput(LSROutput::ChunksPly))
        {
            // Finalize mesh
            lvr2::SimpleFinalizer<BaseVecT> finalize;
            auto meshBuffer = finalize.apply(mesh);

            auto m = ModelPtr(new Model(meshBuffer));
            ModelFactory::saveModel(m, "largeScale_test_" + to_string(voxelSize) + ".ply");
        }
            return mesh;

    }

    template<typename BaseVecT>
    int LargeScaleReconstruction<BaseVecT>::trueMpiAndReconstructMaster(ScanProjectEditMarkPtr project,
    BoundingBox<BaseVecT>& newChunksBB,
            std::shared_ptr<ChunkHashGrid> chunkManager, int size)
    {
        unsigned long timeStart = lvr2::timestamp.getCurrentTimeInMs();
        unsigned long timeInit;
        unsigned long timeCalc;
        unsigned long timeReconstruct;

        float chunkSize = chunkManager->getChunkSize();

        if(project->project->positions.size() != project->changed.size())
        {
            cout << "Inconsistency between number of given scans and diff-vector (scans to consider)! exit..." << endl;
            bool a = false;
            for(int i = 1; i < size; i++)
            {
                MPI_Recv(nullptr, 0, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&a, 1, MPI_CXX_BOOL, i, 0, MPI_COMM_WORLD);
            }
            return 0;
        }

        cout << lvr2::timestamp << "Starting BigGrid" << endl;
        BigGrid<BaseVecT> bg( m_options.bgVoxelSize ,project, m_options.scale);
        cout << lvr2::timestamp << "BigGrid finished " << endl;

        BoundingBox<BaseVecT> bb = bg.getBB();

        std::vector<BoundingBox<BaseVecT>> partitionBoxes;
        BoundingBox<BaseVecT> cmBB = BoundingBox<BaseVecT>();




        BoundingBox<BaseVecT> partbb = bg.getpartialBB();
        cout << lvr2::timestamp << "generating VGrid" << endl;

        VirtualGrid<BaseVecT> vGrid(
                bg.getpartialBB(), chunkSize, m_options.bgVoxelSize);
        vGrid.calculateBoxes();
        partitionBoxes = vGrid.getBoxes();
        BaseVecT addMin = BaseVecT(std::floor(partbb.getMin().x / chunkSize) * chunkSize, std::floor(partbb.getMin().y / chunkSize) * chunkSize, std::floor(partbb.getMin().z / chunkSize) * chunkSize);
        BaseVecT addMax = BaseVecT(std::ceil(partbb.getMax().x / chunkSize) * chunkSize, std::ceil(partbb.getMax().y / chunkSize) * chunkSize, std::ceil(partbb.getMax().z / chunkSize) * chunkSize);
        newChunksBB.expand(addMin);
        newChunksBB.expand(addMax);
        cout << lvr2::timestamp << "finished vGrid" << endl;
        std::cout << lvr2::timestamp << "got: " << partitionBoxes.size() << " Chunks"
                  << std::endl;

        // we use the BB of all scans (including old ones) they are already hashed in the cm
        // and we can't make the BB smaller
        BaseVecT addCMBBMin = BaseVecT(std::floor(bb.getMin().x / chunkSize) * chunkSize, std::floor(bb.getMin().y / chunkSize) * chunkSize, std::floor(bb.getMin().z / chunkSize) * chunkSize);
        BaseVecT addCMBBMax = BaseVecT(std::ceil(bb.getMax().x / chunkSize) * chunkSize, std::ceil(bb.getMax().y / chunkSize) * chunkSize, std::ceil(bb.getMax().z / chunkSize) * chunkSize);
        cmBB.expand(addCMBBMin);
        cmBB.expand(addCMBBMax);

        // TODO: It breaks here

        chunkManager->setBoundingBox(cmBB);
        size_t numChunks_global = (cmBB.getXSize() / chunkSize) * (cmBB.getYSize() / chunkSize) * (cmBB.getZSize() / chunkSize);
        size_t numChunks_partial = partitionBoxes.size();

        cout << lvr2::timestamp << "Saving " << numChunks_global - numChunks_partial << " Chunks compared to full reconstruction" << endl;

        BaseVecT bb_min(bb.getMin().x, bb.getMin().y, bb.getMin().z);
        BaseVecT bb_max(bb.getMax().x, bb.getMax().y, bb.getMax().z);
        BoundingBox<BaseVecT> cbb(bb_min, bb_max);

        timeInit = lvr2::timestamp.getCurrentTimeInMs();

        // TODO: Setting up multithreading for MPI
//        #pragma omp parallel sections
//        {
//            #pragma omp section
//            {
                uint* partitionBoxesSkipped = mpiScheduler(partitionBoxes, bg, cbb, chunkManager);
//                timeCalc = lvr2::timestamp.getCurrentTimeInMs();
//            }
//            #pragma omp section
//            {
                mpiCollector(partitionBoxes, cbb, chunkManager, partitionBoxesSkipped);
//            }
//        }

        delete [] partitionBoxesSkipped;

        unsigned long timeEnd = lvr2::timestamp.getCurrentTimeInMs();

        unsigned long timeSum = timeEnd - timeStart;

        cout << "Finished complete reconstruction in " << (double) (timeSum/1000.0) << "s" << endl;
        cout << "Initialization: " << (double) ((timeInit-timeStart)/1000.0) << "s" << endl;
        cout << "Calculation: " << (double) ((timeCalc-timeInit)/1000.0) << "s" << endl;
        cout << "Combine chunks: " << (double) ((timeEnd-timeCalc)/1000.0) << "s" << endl;


        return 1;

    }

    template<typename BaseVecT>
    uint* LargeScaleReconstruction<BaseVecT>::mpiScheduler(const std::vector<BoundingBox<BaseVecT>>& partitionBoxes, BigGrid<BaseVecT>& bg, BoundingBox<BaseVecT>& cbb, std::shared_ptr<ChunkHashGrid> chunkManager)
    {
        /// Minimum number of points needed to consider a chunk. Chunks smaller than this are skipped.
        size_t minPointsPerChunk = std::max(m_options.ki, std::max(m_options.kd, m_options.kn)) * 2;

        uint* partitionBoxesSkipped = new uint[m_options.voxelSizes.size()]();
        for(int h = 0; h < m_options.voxelSizes.size(); h++)
        {
            float voxelSize = m_options.voxelSizes[h];
            float overlap = 5 * voxelSize;
            BaseVecT overlapVector(overlap, overlap, overlap);

            // send chunks

            for (int i = 0; i < partitionBoxes.size(); i++)
            {
                std::cout << lvr2::timestamp << "Chunk " << i+1 << "/" << partitionBoxes.size() << std::endl;


                size_t numPoints;

                BoundingBox<BaseVecT> gridbb(partitionBoxes[i].getMin() - overlapVector, partitionBoxes[i].getMax() + overlapVector);

                floatArr points = bg.points(gridbb, numPoints, minPointsPerChunk);

                if (!points)
                {
                    partitionBoxesSkipped[h]++;
                    continue;
                }

                // Get Bounding Box
                float x_min = gridbb.getMin().x, y_min = gridbb.getMin().y, z_min = gridbb.getMin().z,
                      x_max = gridbb.getMax().x, y_max = gridbb.getMax().y, z_max = gridbb.getMax().z;


                // Wait for Client to ask for job
                int dest;
                MPI_Status status;
                MPI_Recv(nullptr, 0, MPI_BYTE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                dest = status.MPI_SOURCE;
                std::cout << lvr2::timestamp << "Send chunk to client " << dest << std::endl;
                MPI_Send(&i, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);

                // Send all Data to client
                // TODO: Non-blocking
                std::cout << lvr2::timestamp << "Num Points: " << numPoints << std::endl;
                MPI_Send(&numPoints, 1, MPI_SIZE_T, dest, 3, MPI_COMM_WORLD);
                std::cout << lvr2::timestamp << "Points: " << points.get()[0] << std::endl;
                MPI_Send(points.get(), numPoints*3, MPI_FLOAT, dest, 4, MPI_COMM_WORLD);
                std::cout << lvr2::timestamp << "BoundingBoxMin: [" << x_min << "," << y_min << "," << z_min << "]" << std::endl;
                MPI_Send(&x_min, 1, MPI_FLOAT, dest, 5, MPI_COMM_WORLD);
                MPI_Send(&y_min, 1, MPI_FLOAT, dest, 6, MPI_COMM_WORLD);
                MPI_Send(&z_min, 1, MPI_FLOAT, dest, 7, MPI_COMM_WORLD);
                std::cout << lvr2::timestamp << "BoundingBoxMin: [" << x_max << "," << y_max << "," << z_max << "]" << std::endl;
                MPI_Send(&x_max, 1, MPI_FLOAT, dest, 8, MPI_COMM_WORLD);
                MPI_Send(&y_max, 1, MPI_FLOAT, dest, 9, MPI_COMM_WORLD);
                MPI_Send(&z_max, 1, MPI_FLOAT, dest, 10, MPI_COMM_WORLD);
                std::cout << lvr2::timestamp << "h: " << h << std::endl;
                MPI_Send(&h, 1, MPI_INT, dest, 11, MPI_COMM_WORLD);
                bool calcNorm = !bg.hasNormals();
                std::cout << lvr2::timestamp << "Normals available: " << !calcNorm << std::endl;
                MPI_Send(&calcNorm, 1, MPI_CXX_BOOL, dest, 12, MPI_COMM_WORLD);


                // Send normals if they are available
                if (!calcNorm)
                {
                    size_t numNormals;
                    lvr2::floatArr normals = bg.normals(gridbb, numNormals);
                    std::cout << lvr2::timestamp << "NumNormals: " << numNormals << std::endl;
                    MPI_Send(&numNormals, 1, MPI_SIZE_T, dest, 13, MPI_COMM_WORLD);
                    MPI_Send(normals.get(), numNormals*3, MPI_FLOAT, dest, 14, MPI_COMM_WORLD);
                }
                std::cout << std::endl;
                // Wait for new client
            }
            std::cout << lvr2::timestamp << "Skipped PartitionBoxes: " << partitionBoxesSkipped << std::endl;
        }
        int size;
        int a = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Status status;
        std::cout << lvr2::timestamp << "All chunks send, sending abort codes... " << std::endl;
        for(int i = 1; i < size; i++)
        {
            std::cout << lvr2::timestamp << "Abort " << i << "/" << size-1 << std::endl;
            MPI_Recv(nullptr, 0, MPI_BYTE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            int dest = status.MPI_SOURCE;
            std::cout << lvr2::timestamp << "Sending abort code to client " << dest << std::endl;
            MPI_Send(&a, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
        }
        std::cout << lvr2::timestamp << "All clients closed." << std::endl;
        return partitionBoxesSkipped;
    }

    template<typename BaseVecT>
    void LargeScaleReconstruction<BaseVecT>::mpiCollector(const std::vector<BoundingBox<BaseVecT>>& partitionBoxes, BoundingBox<BaseVecT>& cbb, std::shared_ptr<ChunkHashGrid> chunkManager, uint* partitionBoxesSkipped)
    {
        std::vector<BoundingBox<BaseVecT>> filteredPartitionBoxes;
        unsigned long timeSum = 0;
        float chunkSize = chunkManager->getChunkSize();

        for(int h = 0; h < m_options.voxelSizes.size(); h++)
        {
            float voxelSize = m_options.voxelSizes[h];
            float overlap = 5 * voxelSize;
            BaseVecT overlapVector(overlap, overlap, overlap);

            // vector to save the new chunk names - which chunks have to be reconstructed
            std::vector<BaseVector<int>> newChunks;

            string layerName = "tsdf_values_" + std::to_string(m_options.voxelSizes[h]);

            for (int i = 0; i < partitionBoxes.size() - partitionBoxesSkipped[h]; i++)
            {
                auto& partitionBox = partitionBoxes[i];
                BoundingBox<BaseVecT> gridbb(partitionBox.getMin() - overlapVector, partitionBox.getMax() + overlapVector);

                // Receive chunk from client
                int len, dest, chunk;
                MPI_Status status;
                std::cout << lvr2::timestamp << "[Collector] Waiting for chunk " << i+1 << "/" << partitionBoxes.size() - partitionBoxesSkipped[h] << std::endl;
                MPI_Recv(&len, 1, MPI_INT, MPI_ANY_SOURCE, 15, MPI_COMM_WORLD, &status);
                dest = status.MPI_SOURCE;
                std::cout << lvr2::timestamp << "[Collector] Got chunk from Client " << dest << std::endl << std::endl;
                char* ret = new char[len];
                MPI_Recv(ret, len, MPI_CHAR, dest, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Recv(&chunk, 1, MPI_INT, dest, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                time_t now = time(0);
                tm *time = localtime(&now);
                stringstream largeScale;
                largeScale << 1900 + time->tm_year << "_" << 1+ time->tm_mon << "_" << time->tm_mday << "_" <<  time->tm_hour << "h_" << 1 + time->tm_min << "m_" << 1 + time->tm_sec << "s.dat";
                std::ofstream file(largeScale.str().c_str());
                file.write(ret, len);
                file.close();

                auto ps_grid = std::make_shared<HashGrid<BaseVecT, BoxT>>(largeScale.str(), gridbb, voxelSize);

                std::remove(largeScale.str().c_str());

                delete [] ret;


                unsigned long timeStart = lvr2::timestamp.getCurrentTimeInMs();
                int x = (int) floor(partitionBoxes[chunk].getCentroid().x / chunkSize);
                int y = (int) floor(partitionBoxes[chunk].getCentroid().y / chunkSize);
                int z = (int) floor(partitionBoxes[chunk].getCentroid().z / chunkSize);


                chunkManager->setChunk(layerName, x, y, z, ps_grid->toPointBuffer());
                BaseVector<int> chunkCoordinates(x, y, z);
                // also save the grid coordinates of the chunk added to the ChunkManager
                newChunks.push_back(chunkCoordinates);
                // also save the "real" bounding box without overlap
                filteredPartitionBoxes.push_back(partitionBoxes[chunk]);

                unsigned long timeEnd = lvr2::timestamp.getCurrentTimeInMs();

                timeSum += timeEnd - timeStart;
            }

            cout << "ChunkManagerIO Time: " <<(double) (timeSum / 1000.0) << " s" << endl;
            cout << lvr2::timestamp << "finished" << endl;

            if(h == 0 && m_options.hasOutput(LSROutput::BigMesh))
            {
                //combine chunks
                auto vmax = cbb.getMax() - overlapVector;
                auto vmin = cbb.getMin() + overlapVector;
                cbb.expand(vmin);
                cbb.expand(vmax);

                // auto hg = std::make_shared<HashGrid<BaseVecT, BoxT>>(grid_files, cbb, m_voxelSize);
                // don't read from HDF5 - get the chunks from the ChunkManager
                // auto hg = std::make_shared<HashGrid<BaseVecT, BoxT>>(m_filePath, newChunks, cbb);
                // TODO: don't do the following reconstruction in ChunkingPipline-Workflow (put it in extra function for lsr_tool)
                std::vector<PointBufferPtr> tsdfChunks;
                for (BaseVector<int> coord : newChunks) {
                    boost::optional<shared_ptr<PointBuffer>> chunk = chunkManager->getChunk<PointBufferPtr>(layerName,
                                                                                                            coord.x,
                                                                                                            coord.y,
                                                                                                            coord.z);
                    if (chunk) {
                        tsdfChunks.push_back(chunk.get());
                    } else {
                        std::cout << "WARNING - Could not find chunk (" << coord.x << ", " << coord.y << ", " << coord.z
                                  << ") in layer: " << "tsdf_values_" + std::to_string(voxelSize) << std::endl;
                    }
                }
                auto hg = std::make_shared<HashGrid<BaseVecT, BoxT>>(tsdfChunks, filteredPartitionBoxes, cbb, voxelSize);
                tsdfChunks.clear();

                createAndSaveBigMesh(hg, h);
            }
            std::cout << lvr2::timestamp << "added/changed " << newChunks.size() << " chunks in layer " << layerName << std::endl;
        }
    }

    template<typename BaseVecT>
    int LargeScaleReconstruction<BaseVecT>::trueMpiAndReconstructSlave()
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int chunk;
        std::list<int> chunks;

        std::cout << lvr2::timestamp << "[" << rank << "] Waiting for work." << std::endl;
        MPI_Send(nullptr, 0, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        MPI_Recv(&chunk, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        while(chunk != -1)
        {
            chunks.push_back(chunk);
            float x_min, y_min, z_min, x_max, y_max, z_max;
            size_t numPoints;
            bool calcNorm;
            int h;


            // Receive chunk information
            MPI_Recv(&numPoints, 1, MPI_SIZE_T, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            floatArr points(new float[numPoints*3]);
            MPI_Recv(points.get(), numPoints*3, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&x_min, 1, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&y_min, 1, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&z_min, 1, MPI_FLOAT, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&x_max, 1, MPI_FLOAT, 0, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&y_max, 1, MPI_FLOAT, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&z_max, 1, MPI_FLOAT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&h, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&calcNorm, 1, MPI_CXX_BOOL, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            BoundingBox<BaseVecT> gridbb(BaseVecT(x_min, y_min, z_min), BaseVecT(x_max, y_max, z_max));


            lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);
            p_loader->setPointArray(points, numPoints);

            if (!calcNorm)
            {
              // receive normals if they are available
              size_t numNormals;
              MPI_Recv(&numNormals, 1, MPI_SIZE_T, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              floatArr normals(new float[numNormals]);
              MPI_Recv(normals.get(), numNormals, MPI_FLOAT, 0, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

              p_loader->setNormalArray(normals, numNormals);
            }

            // if(numPoints > (chunkSize*500000)) // reduction TODO add options
            // {
            //     OctreeReduction oct(p_loader, m_voxelSizes[h], 20);
            //     p_loader = oct.getReducedPoints();
            // }

            lvr2::PointsetSurfacePtr<BaseVecT> surface;
            surface = make_shared<lvr2::AdaptiveKSearchSurface<BaseVecT>>(p_loader,
                                                                     "FLANN",
                                                                     m_options.kn,
                                                                     m_options.ki,
                                                                     m_options.kd,
                                                                     m_options.useRansac);

            //calculate important stuff for reconstruction
            if (calcNorm)
            {
                surface->calculateSurfaceNormals();
            }
            auto ps_grid = std::make_shared<lvr2::PointsetGrid<BaseVecT, BoxT>>(
                    m_options.voxelSizes[h], surface, gridbb, true, m_options.extrude);

            ps_grid->calcDistanceValues();

            stringstream largeScale;
            largeScale << "/tmp/lvr2_lsr_mpi_" << rank << "_" << chunk;

            ps_grid->saveGrid(largeScale.str());


//            MPI_Send(&len, 1, MPI_INT, 0, 15 /*+ chunk * 2*/, MPI_COMM_WORLD);
//            MPI_Send(result, len, MPI_CHAR, 0, 16 /*+ chunk * 2*/, MPI_COMM_WORLD);
//            MPI_Send(&chunk, 1, MPI_INT, 0, 17, MPI_COMM_WORLD);



            std::cout << lvr2::timestamp << "[" << rank << "] Requesting chunk. " << std::endl;
            // is something to do?

            MPI_Send(nullptr, 0, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&chunk, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Send result back to master
        for(auto chunk : chunks)
        {
            stringstream largeScale;
            largeScale << "/tmp/lvr2_lsr_mpi_" << rank << "_" << chunk;

            ifstream fl(largeScale.str());
            fl.seekg(0, ios::end);
            int len = fl.tellg();
            char* result = new char[len];
            fl.seekg(0, ios::beg);
            fl.read(result, len);
            fl.close();

            cout << "Sending " << len << " bytes." << endl;
            MPI_Send(&len, 1, MPI_INT, 0, 15 , MPI_COMM_WORLD);
            cout << "Sending data." << endl;
            MPI_Send(result, len, MPI_CHAR, 0, 16 , MPI_COMM_WORLD);
            cout << "Sending chunk " << chunk + 1 << endl;
            MPI_Send(&chunk, 1, MPI_INT, 0, 17, MPI_COMM_WORLD);
            std::remove(largeScale.str().c_str());
            delete [] result;
        }
        std::cout << lvr2::timestamp << "[" << rank << "] finished. " << std::endl;
        return 1;
    }

}
