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
#include "lvr2/algorithm/HLODTree.hpp"
#include "lvr2/algorithm/pmp/SurfaceNormals.h"
#include "lvr2/io/baseio/ChannelIO.hpp"
#include "lvr2/io/baseio/ArrayIO.hpp"
#include "lvr2/io/baseio/VariantChannelIO.hpp"
#include "lvr2/io/Tiles3dIO.hpp"
#include "lvr2/reconstruction/VirtualGrid.hpp"
#include "lvr2/reconstruction/BigGridKdTree.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/FastBox.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/registration/OctreeReduction.hpp"

#include "lvr2/algorithm/CleanupAlgorithms.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/algorithm/Tesselator.hpp"
#include "lvr2/util/Timestamp.hpp"


#include "LargeScaleReconstruction.hpp"

#include <mpi.h>
#include <yaml-cpp/yaml.h>


#if defined CUDA_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/cuda/CudaSurface.hpp"

    typedef lvr2::CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/opencl/ClSurface.hpp"
    typedef lvr2::ClSurface GpuSurface;
#endif

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

    using Vec = lvr2::BaseVector<float>;

    //TODO: write set Methods for certain variables

    template<typename BaseVecT>
    LargeScaleReconstruction<BaseVecT>::LargeScaleReconstruction(LSROptions options)
        : m_options(options)
    {
#ifndef GPU_FOUND
        if (m_options.useGPU)
        {
            std::cout << timestamp << "Warning: useGPU specified but no GPU found. Reverting to CPU" << std::endl;
            m_options.useGPU = false;
        }
#endif

        std::cout << timestamp << "Reconstruction Instance generated..." << std::endl;
    }

    template <typename BaseVecT>
    void LargeScaleReconstruction<BaseVecT>::chunkAndReconstruct(
        ScanProjectEditMarkPtr project,
        BoundingBox<BaseVecT>& newChunksBB,
        std::shared_ptr<ChunkHashGrid> chunkManager)
    {
#ifndef GPU_FOUND
        if (m_options.useGPU)
        {
            std::cout << timestamp << "Warning: useGPU specified but no GPU found. Reverting to CPU" << std::endl;
            m_options.useGPU = false;
        }
#endif

        auto startTimeMs = timestamp.getCurrentTimeInMs();

        if(project->project->positions.size() != project->changed.size())
        {
            throw std::runtime_error("Inconsistency between number of given scans and diff-vector (scans to consider)!");
        }

        bool useVGrid = m_options.partMethod == 1;
        if (useVGrid && !chunkManager)
        {
            throw std::runtime_error("partMethod set to VGrid but no chunk manager given!");
        }

        pmp::Point flipPoint(m_options.flipPoint[0], m_options.flipPoint[1], m_options.flipPoint[2]);

        /// Minimum number of points needed to consider a chunk. Chunks smaller than this are skipped.
        size_t minPointsPerChunk = std::max(m_options.ki, std::max(m_options.kd, m_options.kn)) * 2;

        /// Maximum number of points in a chunk to avoid GPU memory overflow. Chunks bigger than this are reduced.
        size_t maxPointsPerChunk = 130'000'000;

        float chunkSize = m_options.bgVoxelSize;

        std::cout << timestamp << "Starting BigGrid" << std::endl;
        BigGrid<BaseVecT> bg(chunkSize, project, m_options.scale, m_options.extrude);
        std::cout << timestamp << "BigGrid finished " << std::endl;

        BoundingBox<BaseVecT> bgBB = bg.getBB();

        std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes;
        vector<BoundingBox<BaseVecT>> partitionBoxesNew;
        vector<BaseVector<int>> partitionChunkCoords;

        if (useVGrid)
        {
            // BigGrid is essentially a V-Grid that already knows where all the points are,
            // so just recycle the cells as partitions.

            auto& cells = bg.getCells();
            partitionBoxes.reset(new vector<BoundingBox<BaseVecT>>());
            partitionBoxes->reserve(cells.size());

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
                partitionBoxes->push_back(bb);
                partitionChunkCoords.emplace_back(index.x(), index.y(), index.z());
            }
            cmBB.expand(newChunksBB);
            chunkManager->setBoundingBox(cmBB);
        }
        else
        {
            // use KD-Tree
            std::cout << timestamp << "generating tree" << std::endl;
            BigGridKdTree<BaseVecT> gridKd(bg.getBB(), m_options.nodeSize, &bg, m_options.bgVoxelSize);
            gridKd.insert(bg.pointSize(), bg.getBB().getCentroid());
            ofstream partBoxOfs("KdTree.ser");
            partitionBoxes = shared_ptr<vector<BoundingBox<BaseVecT>>>(new vector<BoundingBox<BaseVecT>>(gridKd.getLeafs().size()));
            for (size_t i = 0; i < gridKd.getLeafs().size(); i++)
            {
                BoundingBox<BaseVecT> partBB = gridKd.getLeafs()[i]->getBB();
                partitionBoxes->at(i) = partBB;
                partBoxOfs << partBB.getMin()[0] << " " << partBB.getMin()[1] << " "
                        << partBB.getMin()[2] << " " << partBB.getMax()[0] << " "
                        << partBB.getMax()[1] << " " << partBB.getMax()[2] << std::endl;
            }

            std::cout << timestamp << "Finished tree" << std::endl;
        }

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
            float overlap = chunkSize / 2.0f;
            overlap = std::ceil(overlap / voxelSize) * voxelSize; // make sure overlap is divisible by voxelSize
            BaseVecT overlapVector(overlap, overlap, overlap);

            bool createBigMesh = false, createChunksPly = false, createChunksHdf5 = false, create3dTiles = false;
            // only produce output for the first voxel size
            if (h == 0)
            {
                createBigMesh = m_options.hasOutput(LSROutput::BigMesh);
                createChunksPly = chunkManager && m_options.hasOutput(LSROutput::ChunksPly);
                createChunksHdf5 = chunkManager && m_options.hasOutput(LSROutput::ChunksHdf5);

#ifdef LVR2_USE_3DTILES
                create3dTiles = chunkManager && m_options.hasOutput(LSROutput::Tiles3d);
#endif
            }

            // chunk map for LSROutput::Tiles3d
            std::unordered_map<Vector3i, typename HLODTree<BaseVecT>::Ptr> chunkMap;

            // file to store chunks in for LSROutput::ChunksHdf5; temp file for LSROutput::Tiles3d
            std::shared_ptr<HighFive::File> chunkFile = nullptr;
            if (createChunksHdf5 || create3dTiles)
            {
                chunkFile = hdf5util::open("chunks.h5", HighFive::File::Truncate);
                auto root = chunkFile->getGroup("/");

                hdf5util::setAttribute(root, "chunk_size", chunkSize);
                hdf5util::setAttribute(root, "voxel_size", voxelSize);
            }
            if (createChunksPly)
            {
                boost::filesystem::remove_all("chunks");
                boost::filesystem::create_directories("chunks");

                YAML::Node metadata;
                metadata["chunk_size"] = chunkSize;
                metadata["voxel_size"] = voxelSize;

                std::ofstream out("chunks/chunk_metadata.yaml");
                out << metadata;
            }

            // V-Grid: vector to save the new chunk names - which chunks have to be reconstructed
            vector<BaseVector<int>> newChunks;

            // KD-Tree: vector to store relevant chunks as .ser
            vector<string> grid_files;

            string layerName = "tsdf_values_" + std::to_string(voxelSize);

            size_t partitionBoxesSkipped = 0;

            for (size_t i = 0; i < partitionBoxes->size(); i++)
            {
                auto& partitionBox = partitionBoxes->at(i);

                BoundingBox<BaseVecT> gridbb(partitionBox.getMin() - overlapVector, partitionBox.getMax() + overlapVector);

                if (bg.estimateSizeofBox(gridbb) < minPointsPerChunk)
                {
                    partitionBoxesSkipped++;
                    continue;
                }

                size_t numPoints;
                floatArr points = bg.points(gridbb, numPoints, minPointsPerChunk);

                if (!points)
                {
                    partitionBoxesSkipped++;
                    continue;
                }

                std::cout << timestamp << "Processing Partition " << i << "/" << (partitionBoxes->size() - 1) << std::endl;

                string name_id;
                if (useVGrid)
                {
                    auto& chunkCoords = partitionChunkCoords[i];
                    name_id = std::to_string(chunkCoords.x) + "_" + std::to_string(chunkCoords.y) + "_" + std::to_string(chunkCoords.z);
                }
                else
                {
                    name_id = std::to_string(i);
                }

                lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);
                p_loader->setPointArray(points, numPoints);

                bool hasNormals = false;
                bool hasDistances = false;

                if (bg.hasNormals())
                {
                    size_t numNormals;
                    lvr2::floatArr normals = bg.normals(gridbb, numNormals);
                    if (numNormals != numPoints)
                    {
                        std::cerr << "ERROR: Number of normals does not match number of points" << std::endl;
                    }
                    else
                    {
                        p_loader->setNormalArray(normals, numNormals);
                        hasNormals = true;
                    }
                }

                if (m_options.useGPU)
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

                lvr2::PointsetSurfacePtr<Vec> surface =
                    make_shared<lvr2::AdaptiveKSearchSurface<Vec>>(p_loader, "FLANN", m_options.kn, m_options.ki, m_options.kd, m_options.useRansac);

                auto ps_grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(voxelSize, surface, gridbb, true, m_options.extrude);

                if (!hasNormals && m_options.useGPU)
                {
                    // std::vector<float> flipPoint = std::vector<float>{100, 100, 100};
                    floatArr points = p_loader->getPointArray();
                    floatArr normals = floatArr(new float[numPoints * 3]);
                    std::cout << timestamp << "Generate GPU kd-tree..." << std::endl;

                    GpuSurface gpu_surface(points, numPoints);

                    gpu_surface.setKn(m_options.kn);
                    gpu_surface.setKi(m_options.ki);
                    gpu_surface.setKd(m_options.kd);
                    gpu_surface.setFlippoint(m_options.flipPoint[0], m_options.flipPoint[1], m_options.flipPoint[2]);

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
                        i--; // retry this partition
                        continue;
                    }
                    gpu_surface.getNormals(normals);

                    p_loader->setNormalArray(normals, numPoints);
                    hasNormals = true;

                    if(m_options.useGPUDistances)
                    {
                        auto& query_points = ps_grid->getQueryPoints();

                        std::cout << timestamp << "Computing signed distances in GPU with brute force kernel (kd = 5)." << std::endl;
                        std::cout << timestamp << "This might take a while...." << std::endl;
                        gpu_surface.distances(query_points, voxelSize);
                        hasDistances = true;
                        std::cout << timestamp << "Done." << std::endl;
                    }
                }

                if (!hasNormals)
                {
                    surface->calculateSurfaceNormals();
                }
                if (!hasDistances)
                {
                    ps_grid->calcDistanceValues();
                }

                if (chunkManager)
                {
                    auto& chunkCoords = partitionChunkCoords[i];
                    addTSDFChunkManager(chunkCoords.x, chunkCoords.y, chunkCoords.z, ps_grid, chunkManager, layerName);
                    // also save the grid coordinates of the chunk added to the ChunkManager
                    newChunks.push_back(chunkCoords);
                }
                else
                {
                    std::stringstream ss2;
                    ss2 << name_id << ".ser";
                    ps_grid->saveCells(ss2.str());
                    grid_files.push_back(ss2.str());
                }

                // also save the "real" bounding box without overlap
                partitionBoxesNew.push_back(partitionBox);

                // save the mesh of the chunk
                if (!createChunksHdf5 && !createChunksPly && !create3dTiles)
                {
                    continue;
                }

                auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(ps_grid);
                lvr2::PMPMesh<Vec> mesh;
                reconstruction->getMesh(mesh);

                if (mesh.numFaces() == 0)
                {
                    continue;
                }

                // run cleanup on the mesh.
                // cleanContours and optimizePlanes might mess with the chunk borders

                if (m_options.removeDanglingArtifacts)
                {
                    cout << timestamp << "Removing dangling artifacts" << endl;
                    removeDanglingCluster(mesh, m_options.removeDanglingArtifacts);
                    if (mesh.numFaces() == 0)
                    {
                        continue;
                    }
                }
                if (m_options.fillHoles)
                {
                    mesh.fillHoles(m_options.fillHoles);
                }

                bool createOtherChunks = createChunksHdf5 || createChunksPly;
                std::unique_ptr<PMPMesh<BaseVecT>> tiles3dMesh = nullptr;
                std::unique_ptr<PMPMesh<BaseVecT>> otherChunksMesh = nullptr;
                if (create3dTiles && createOtherChunks)
                {
                    // tiles3d needs a different mesh than other chunks => create copy
                    tiles3dMesh.reset(new PMPMesh<BaseVecT>(mesh));
                    otherChunksMesh.reset(new PMPMesh<BaseVecT>(std::move(mesh)));
                }
                else if (create3dTiles)
                {
                    tiles3dMesh.reset(new PMPMesh<BaseVecT>(std::move(mesh)));
                }
                else
                {
                    otherChunksMesh.reset(new PMPMesh<BaseVecT>(std::move(mesh)));
                }

                pmp::Point oneVoxel = pmp::Point::Constant(voxelSize);
                pmp::Point epsilon = pmp::Point::Constant(0.0001);
                auto min = partitionBox.getMin(), max = partitionBox.getMax();
                pmp::BoundingBox expectedBB(pmp::Point(min.x, min.y, min.z), pmp::Point(max.x, max.y, max.z));
                expectedBB.min() += oneVoxel / 2 - epsilon;
                expectedBB.max() += oneVoxel / 2 + epsilon;
                if (create3dTiles)
                {
                    // completely trim overlap
                    HLODTree<BaseVecT>::trimChunkOverlap(*tiles3dMesh, expectedBB);
                    if (tiles3dMesh->numFaces() > 0)
                    {
                        pmp::SurfaceNormals::compute_vertex_normals(tiles3dMesh->getSurfaceMesh(), flipPoint);

                        auto& chunkCoords = partitionChunkCoords[i];
                        Vector3i chunkPos(chunkCoords.x, chunkCoords.y, chunkCoords.z);
                        auto bb = tiles3dMesh->getSurfaceMesh().bounds();
                        chunkMap.emplace(chunkPos, HLODTree<BaseVecT>::leaf(LazyMesh(std::move(*tiles3dMesh), chunkFile), bb));
                    }
                }

                if (createOtherChunks)
                {
                    // trim the overlap to only a single voxel to save space
                    expectedBB.min() -= oneVoxel;
                    expectedBB.max() += oneVoxel;
                    HLODTree<BaseVecT>::trimChunkOverlap(*otherChunksMesh, expectedBB);

                    pmp::SurfaceMesh& surfaceMesh = otherChunksMesh->getSurfaceMesh();
                    if (surfaceMesh.n_faces() > 0)
                    {
                        surfaceMesh.remove_vertex_property<bool>("v:feature");
                        surfaceMesh.remove_edge_property<bool>("e:feature");
                        if (createChunksHdf5)
                        {
                            auto group = chunkFile->createGroup("/chunks/" + name_id);
                            surfaceMesh.write(group);
                            chunkFile->flush();
                        }
                        if (createChunksPly)
                        {
                            surfaceMesh.write("chunks/" + name_id + ".ply");
                        }
                    }
                }
            }
            std::cout << timestamp << "Skipped PartitionBoxes: " << partitionBoxesSkipped << std::endl;

            std::cout << timestamp << "finished" << std::endl;

            if(createBigMesh)
            {
                //combine chunks
                BoundingBox<BaseVecT> cbb(bgBB.getMin() - overlapVector, bgBB.getMax() + overlapVector);

                std::shared_ptr<HashGrid<BaseVecT, lvr2::FastBox<Vec>>> hg;

                if (chunkManager)
                {
                    std::vector<PointBufferPtr> tsdfChunks;
                    for (BaseVector<int> coord : newChunks)
                    {
                        auto chunk = chunkManager->getChunk<PointBufferPtr>(layerName, coord.x, coord.y, coord.z);
                        if (chunk)
                        {
                            tsdfChunks.push_back(chunk.get());
                        }
                        else
                        {
                            std::cout << "WARNING - Could not find chunk (" << coord.x << ", " << coord.y << ", " << coord.z
                                    << ") in layer: " << "tsdf_values_" + std::to_string(voxelSize) << std::endl;
                        }
                    }
                    hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(tsdfChunks, partitionBoxesNew, cbb, voxelSize);
                }
                else
                {
                    hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(grid_files, partitionBoxesNew, cbb, voxelSize);
                }

                auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(hg);

                lvr2::PMPMesh<Vec> mesh;

                reconstruction->getMesh(mesh);

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
                        Tesselator<Vec>::apply(mesh, clusterBiMap, faceNormals, m_options.lineFusionThreshold);
                    }
                }

                if (mesh.numFaces() > 0)
                {
                    // Finalize mesh
                    auto meshBuffer = mesh.toMeshBuffer();

                    time_t now = time(0);

                    tm *time = localtime(&now);
                    stringstream largeScale;
                    largeScale << 1900 + time->tm_year << "_" << 1+ time->tm_mon << "_" << time->tm_mday << "_" <<  time->tm_hour << "h_" << 1 + time->tm_min << "m_" << 1 + time->tm_sec << "s.ply";

                    auto m = ModelPtr(new Model(meshBuffer));
                    ModelFactory::saveModel(m, largeScale.str());
                }
                else
                {
                    std::cout << "Warning: Mesh is empty!" << std::endl;
                }
            }

#ifdef LVR2_USE_3DTILES
            if (create3dTiles && !chunkMap.empty())
            {
                std::cout << timestamp << "Creating 3D Tiles: Generating HLOD Tree" << std::endl;
                auto tree = HLODTree<BaseVecT>::partition(std::move(chunkMap), 2);
                tree->finalize(true);

                std::cout << timestamp << "Creating 3D Tiles: Writing to file" << std::endl;
                Tiles3dIO<BaseVecT> io("mesh.3dtiles");
                io.write(tree, m_options.tiles3dCompress);
                tree.reset();

                if (!createChunksHdf5)
                {
                    // file was only created for 3d tiles => delete it
                    std::string filename = chunkFile->getName();
                    chunkFile.reset();
                    boost::filesystem::remove(filename);
                }
                else
                {
                    LazyMesh<BaseVecT>::removeTempDir(chunkFile);
                }
                std::cout << timestamp << "Creating 3D Tiles: Finished" << std::endl;
            }
#endif // LVR2_USE_3DTILES

            std::cout << lvr2::timestamp << "added/changed " << newChunks.size() << " chunks in layer " << layerName << std::endl;
        }

        auto timeDiffMs = lvr2::timestamp.getCurrentTimeInMs() - startTimeMs;

        cout << "Finished complete reconstruction in " << (timeDiffMs / 1000.0) << "s" << endl;
    }


    template <typename BaseVecT>
    void LargeScaleReconstruction<BaseVecT>::addTSDFChunkManager(int x, int y, int z,
            std::shared_ptr<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>> ps_grid, std::shared_ptr<ChunkHashGrid> cm,
            std::string layerName)
    {
        size_t counter = 0;
        size_t csize = ps_grid->getNumberOfCells();
        vector<QueryPoint<BaseVecT>>& qp = ps_grid->getQueryPoints();
        boost::shared_array<float> centers(new float[3 * csize]);
        //cant save bool?
        boost::shared_array<int> extruded(new int[csize]);
        boost::shared_array<float> queryPoints(new float[8 * csize]);

        for(auto it = ps_grid->firstCell() ; it!= ps_grid->lastCell(); it++)
        {
            for (int j = 0; j < 3; ++j)
            {
                centers[3 * counter + j] = it->second->getCenter()[j];
            }

            extruded[counter] = it->second->m_extruded;

            for (int k = 0; k < 8; ++k)
            {
                queryPoints[8 * counter + k] = qp[it->second->getVertex(k)].m_distance;
            }
            ++counter;
        }

        PointBufferPtr chunk = PointBufferPtr(new PointBuffer(centers, csize));
        chunk->addFloatChannel(queryPoints, "tsdf_values", csize, 8);
        chunk->addChannel(extruded, "extruded", csize, 1);
        chunk->addAtomic<unsigned int>(csize, "num_voxel");

        cm->setChunk<PointBufferPtr>(layerName, x, y, z, chunk);
    }


    template<typename BaseVecT>
    HalfEdgeMesh<BaseVecT> LargeScaleReconstruction<BaseVecT>::getPartialReconstruct(BoundingBox<BaseVecT> newChunksBB,
                                                                            std::shared_ptr<ChunkHashGrid> chunkHashGrid,
                                                                            float voxelSize) {
        string layerName = "tsdf_values_" + std::to_string(voxelSize);
        float chunkSize = chunkHashGrid->getChunkSize();

        std::vector<PointBufferPtr> tsdfChunks;
        std::vector<BoundingBox<BaseVecT>> partitionBoxesNew = std::vector<BoundingBox<BaseVecT>>();
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
                        partitionBoxesNew.push_back(temp);
                        completeBB.expand(temp);

                        tsdfChunks.push_back(chunk.get());
                    }
                }
            }
        }


        auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(tsdfChunks, partitionBoxesNew, completeBB,
                                                                           voxelSize);
        tsdfChunks.clear();
        auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(hg);

        lvr2::HalfEdgeMesh<Vec> mesh;
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
            lvr2::SimpleFinalizer<Vec> finalize;
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

        std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes;
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
        std::cout << lvr2::timestamp << "got: " << partitionBoxes->size() << " Chunks"
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
        size_t numChunks_partial = partitionBoxes->size();

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
    uint* LargeScaleReconstruction<BaseVecT>::mpiScheduler(std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes, BigGrid<BaseVecT>& bg, BoundingBox<BaseVecT>& cbb, std::shared_ptr<ChunkHashGrid> chunkManager)
    {
        /// Minimum number of points needed to consider a chunk. Chunks smaller than this are skipped.
        size_t minPointsPerChunk = std::max(m_options.ki, std::max(m_options.kd, m_options.kn)) * 2;

        uint* partitionBoxesSkipped = new uint[m_options.voxelSizes.size()]();
        for(int h = 0; h < m_options.voxelSizes.size(); h++)
        {
            float voxelSize = m_options.voxelSizes[h];
            float overlap = 20 * voxelSize;
            BaseVecT overlapVector(overlap, overlap, overlap);

            // send chunks

            for (int i = 0; i < partitionBoxes->size(); i++)
            {
                std::cout << lvr2::timestamp << "Chunk " << i+1 << "/" << partitionBoxes->size() << std::endl;


                size_t numPoints;

                BoundingBox<BaseVecT> gridbb(partitionBoxes->at(i).getMin() - overlapVector, partitionBoxes->at(i).getMax() + overlapVector);

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
    void LargeScaleReconstruction<BaseVecT>::mpiCollector(std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes, BoundingBox<BaseVecT>& cbb, std::shared_ptr<ChunkHashGrid> chunkManager, uint* partitionBoxesSkipped)
    {
        vector<BoundingBox<BaseVecT>> partitionBoxesNew;
        unsigned long timeSum = 0;
        float chunkSize = chunkManager->getChunkSize();

        for(int h = 0; h < m_options.voxelSizes.size(); h++)
        {
            // vector to save the new chunk names - which chunks have to be reconstructed
            vector<BaseVector<int>> newChunks = vector<BaseVector<int>>();

            string layerName = "tsdf_values_" + std::to_string(m_options.voxelSizes[h]);

            for (int i = 0; i < partitionBoxes->size() - partitionBoxesSkipped[h]; i++)
            {
                // Receive chunk from client
                int len, dest, chunk;
                MPI_Status status;
                std::cout << lvr2::timestamp << "[Collector] Waiting for chunk " << i+1 << "/" << partitionBoxes->size() - partitionBoxesSkipped[h] << std::endl;
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

                auto ps_grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(largeScale.str().c_str());

                std::remove(largeScale.str().c_str());

                delete [] ret;


                unsigned long timeStart = lvr2::timestamp.getCurrentTimeInMs();
                int x = (int) floor(partitionBoxes->at(chunk).getCentroid().x / chunkSize);
                int y = (int) floor(partitionBoxes->at(chunk).getCentroid().y / chunkSize);
                int z = (int) floor(partitionBoxes->at(chunk).getCentroid().z / chunkSize);


                addTSDFChunkManager(x, y, z, ps_grid, chunkManager, layerName);
                BaseVector<int> chunkCoordinates(x, y, z);
                // also save the grid coordinates of the chunk added to the ChunkManager
                newChunks.push_back(chunkCoordinates);
                // also save the "real" bounding box without overlap
                partitionBoxesNew.push_back(partitionBoxes->at(chunk));

                unsigned long timeEnd = lvr2::timestamp.getCurrentTimeInMs();

                timeSum += timeEnd - timeStart;
            }

            cout << "ChunkManagerIO Time: " <<(double) (timeSum / 1000.0) << " s" << endl;
            cout << lvr2::timestamp << "finished" << endl;

            if(h == 0 && m_options.hasOutput(LSROutput::BigMesh))
            {
                //combine chunks
                float voxelSize = m_options.voxelSizes[h];
                float overlap = 20 * voxelSize;
                BaseVecT overlapVector(overlap, overlap, overlap);
                auto vmax = cbb.getMax() - overlapVector;
                auto vmin = cbb.getMin() + overlapVector;
                cbb.expand(vmin);
                cbb.expand(vmax);

                // auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(grid_files, cbb, m_voxelSize);
                // don't read from HDF5 - get the chunks from the ChunkManager
                // auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(m_filePath, newChunks, cbb);
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
                auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(tsdfChunks, partitionBoxesNew, cbb, voxelSize);
                tsdfChunks.clear();
                auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(hg);

                lvr2::HalfEdgeMesh<Vec> mesh;

                reconstruction->getMesh(mesh);

                if (m_options.removeDanglingArtifacts) {
                    cout << timestamp << "Removing dangling artifacts" << endl;
                    removeDanglingCluster(mesh, static_cast<size_t>(m_options.removeDanglingArtifacts));
                }

                // Magic number from lvr1 `cleanContours`...
                cleanContours(mesh, m_options.cleanContours, 0.0001);

                // Fill small holes if requested
                if (m_options.fillHoles) {
                    naiveFillSmallHoles(mesh, m_options.fillHoles, false);
                }

                // Calculate normals for vertices
                auto faceNormals = calcFaceNormals(mesh);

                ClusterBiMap<FaceHandle> clusterBiMap;
                if (m_options.optimizePlanes) {
                    clusterBiMap = iterativePlanarClusterGrowing(mesh,
                                                                 faceNormals,
                                                                 m_options.planeNormalThreshold,
                                                                 m_options.planeIterations,
                                                                 m_options.minPlaneSize);

                    if (m_options.smallRegionThreshold > 0) {
                        deleteSmallPlanarCluster(
                                mesh, clusterBiMap, static_cast<size_t>(m_options.smallRegionThreshold));
                    }

                    if (m_options.retesselate) {
                        Tesselator<Vec>::apply(
                                mesh, clusterBiMap, faceNormals, m_options.lineFusionThreshold);
                    }
                } else {
                    clusterBiMap = planarClusterGrowing(mesh, faceNormals, m_options.planeNormalThreshold);
                }



                // Finalize mesh
                lvr2::SimpleFinalizer<Vec> finalize;
                auto meshBuffer = finalize.apply(mesh);

                time_t now = time(0);

                tm *time = localtime(&now);
                stringstream largeScale;
                largeScale << 1900 + time->tm_year << "_" << 1+ time->tm_mon << "_" << time->tm_mday << "_" <<  time->tm_hour << "h_" << 1 + time->tm_min << "m_" << 1 + time->tm_sec << "s.ply";

                auto m = ModelPtr(new Model(meshBuffer));
                ModelFactory::saveModel(m, largeScale.str());
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

            lvr2::PointsetSurfacePtr<Vec> surface;
            surface = make_shared<lvr2::AdaptiveKSearchSurface<Vec>>(p_loader,
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
            auto ps_grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(
                    m_options.voxelSizes[h], surface, gridbb, true, m_options.extrude);

            ps_grid->setBB(gridbb);
            ps_grid->calcIndices();
            ps_grid->calcDistanceValues();

            stringstream largeScale;
            largeScale << "/tmp/lvr2_lsr_mpi_" << rank << "_" << chunk;

            ps_grid->serialize(largeScale.str());


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
