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
#include "lvr2/io/baseio/ChannelIO.hpp"
#include "lvr2/io/baseio/ArrayIO.hpp"
#include "lvr2/io/baseio/VariantChannelIO.hpp"
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

    template <typename BaseVecT>
    LargeScaleReconstruction<BaseVecT>::LargeScaleReconstruction()
    : m_voxelSizes(std::vector<float>{0.1}), m_bgVoxelSize(1), m_scale(1),m_nodeSize(1000000), m_partMethod(1),
    m_ki(20), m_kd(25), m_kn(20), m_useRansac(false), m_flipPoint(std::vector<float>{10000000, 10000000, 10000000}), m_extrude(false), m_removeDanglingArtifacts(0), m_cleanContours(0),
    m_fillHoles(0), m_optimizePlanes(false), m_planeNormalThreshold(0.85), m_planeIterations(3), m_minPlaneSize(7), m_smallRegionThreshold(0),
    m_retesselate(false), m_lineFusionThreshold(0.01)
    {
        std::cout << timestamp << "Reconstruction Instance generated..." << std::endl;
    }

    template<typename BaseVecT>
    LargeScaleReconstruction<BaseVecT>::LargeScaleReconstruction( vector<float> voxelSizes, float bgVoxelSize,
                                                                 float scale, uint nodeSize,
                                                                 int partMethod, int ki, int kd, int kn, 
                                                                 bool useRansac,
                                                                 std::vector<float> flipPoint,
                                                                 bool extrude, 
                                                                 int removeDanglingArtifacts,
                                                                 int cleanContours, int fillHoles, 
                                                                 bool optimizePlanes,
                                                                 float planeNormalThreshold, 
                                                                 int planeIterations,
                                                                 int minPlaneSize, 
                                                                 int smallRegionThreshold,
                                                                 bool retesselate, 
                                                                 float lineFusionThreshold,
                                                                 bool bigMesh, 
                                                                 bool debugChunks, 
                                                                 bool useGPU, 
                                                                 bool useGPUDistances)
            : m_voxelSizes(voxelSizes), m_bgVoxelSize(bgVoxelSize),
              m_scale(scale),m_nodeSize(nodeSize),
              m_partMethod(partMethod), 
              m_ki(ki), m_kd(kd), m_kn(kn), 
              m_useRansac(useRansac),
              m_flipPoint(flipPoint), 
              m_extrude(extrude), 
              m_removeDanglingArtifacts(removeDanglingArtifacts),
              m_cleanContours(cleanContours), 
              m_fillHoles(fillHoles), 
              m_optimizePlanes(optimizePlanes),
              m_planeNormalThreshold(planeNormalThreshold), 
              m_planeIterations(planeIterations),
              m_minPlaneSize(minPlaneSize), 
              m_smallRegionThreshold(smallRegionThreshold),
              m_retesselate(retesselate), 
              m_lineFusionThreshold(lineFusionThreshold),
              m_bigMesh(bigMesh), 
              m_debugChunks(debugChunks), 
              m_useGPU(useGPU),
              m_useGPUDistances(useGPUDistances)
    {
        std::cout << timestamp << "Reconstruction Instance generated..." << std::endl;
    }

    template<typename BaseVecT>
    LargeScaleReconstruction<BaseVecT>::LargeScaleReconstruction(LSROptions options)
            : LargeScaleReconstruction(
              options.voxelSizes, 
              options.bgVoxelSize,
              options.scale,options.nodeSize,
              options.partMethod, 
              options.ki, options.kd, options.kn, 
              options.useRansac,
              options.getFlipPoint(), 
              options.extrude, 
              options.removeDanglingArtifacts,
              options.cleanContours, 
              options.fillHoles, 
              options.optimizePlanes,
              options.planeNormalThreshold, 
              options.planeIterations,
              options.minPlaneSize, 
              options.smallRegionThreshold,
              options.retesselate, 
              options.lineFusionThreshold, 
              options.bigMesh, 
              options.debugChunks, 
              options.useGPU, 
              options.useGPUDistances)
    {
    }


    template<typename BaseVecT>
    int LargeScaleReconstruction<BaseVecT>::mpiAndReconstruct(ScanProjectEditMarkPtr project){

        if(project->project->positions.size() != project->changed.size())
        {
            cout << "Inconsistency between number of given scans and diff-vector (scans to consider)! exit..." << endl;
            return 0;
        }

        cout << lvr2::timestamp << "Starting BigGrid" << endl;
        BigGrid<BaseVecT> bg( m_bgVoxelSize ,project, m_scale);
        cout << lvr2::timestamp << "BigGrid finished " << endl;

        BoundingBox<BaseVecT> bb = bg.getBB();

        std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes;
        vector<BoundingBox<BaseVecT>> partitionBoxesNew;

        BaseVecT bb_min(bb.getMin().x, bb.getMin().y, bb.getMin().z);
        BaseVecT bb_max(bb.getMax().x, bb.getMax().y, bb.getMax().z);
        BoundingBox<BaseVecT> cbb(bb_min, bb_max);

        cout << lvr2::timestamp << "generating tree" << endl;
        BigGridKdTree<BaseVecT> gridKd(bg.getBB(), m_nodeSize, &bg, m_bgVoxelSize);
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

        cout << lvr2::timestamp << "Finished tree" << endl;
        std::cout << lvr2::timestamp << "Got " << partitionBoxes->size() << " leafs." << std::endl;
        std::cout << lvr2::timestamp << "Saving leafs..." << std::endl;

        uint partitionBoxesSkipped = 0;

        for(int h = 0; h < m_voxelSizes.size(); h++)
        {
            //vector to store relevant chunks as .ser
            vector<string> grid_files;
            for (int i = 0; i < partitionBoxes->size(); i++)
            {
                string name_id;
                name_id = std::to_string(i);

                size_t numPoints;

                BaseVecT gridbb_min = partitionBoxes->at(i).getMin();
                BaseVecT gridbb_max = partitionBoxes->at(i).getMax();
                for (int j = 0; j < 3; j++)
                {
                    gridbb_min[j] -= m_voxelSizes[h] * 3;
                    gridbb_max[j] += m_voxelSizes[h] * 3;
                }

                floatArr points = bg.points(gridbb_min, gridbb_max, numPoints);

                // remove boxes with less than 50 points
                if (numPoints <= 50)
                {
                    partitionBoxesSkipped++;
                    continue;
                }

                BoundingBox<BaseVecT> gridbb(gridbb_min, gridbb_max);

                cout << "\n" <<  lvr2::timestamp <<"box: " << i << "/" << partitionBoxes->size() - 1 << endl;

                lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);
                p_loader->setPointArray(points, numPoints);

                if (bg.hasNormals())
                {
                    size_t numNormals;
                    lvr2::floatArr normals = bg.normals(gridbb_min, gridbb_max, numNormals);

                    p_loader->setNormalArray(normals, numNormals);
                    cout << "got " << numNormals << " normals" << endl;
                }

                lvr2::PointBufferPtr p_loader_reduced;
                //if(numPoints > (m_chunkSize*500000)) // reduction TODO add options
                if(false)
                {
                    OctreeReduction oct(p_loader, m_voxelSizes[h], 20);
                    p_loader_reduced = oct.getReducedPoints();
                }
                else
                {
                    p_loader_reduced = p_loader;
                }

                lvr2::PointsetSurfacePtr<Vec> surface;
                surface = make_shared<lvr2::AdaptiveKSearchSurface<Vec>>(p_loader_reduced,
                                                                         "FLANN",
                                                                         m_kn,
                                                                         m_ki,
                                                                         m_kd,
                                                                         m_useRansac);
                //calculate important stuff for reconstruction
                if (!bg.hasNormals())
                {
                    if (m_useGPU)
                    {
#ifdef GPU_FOUND
                        // std::vector<float> flipPoint = std::vector<float>{100, 100, 100};
                        size_t num_points = p_loader_reduced->numPoints();
                        floatArr points = p_loader_reduced->getPointArray();
                        floatArr normals = floatArr(new float[num_points * 3]);
                        std::cout << timestamp << "Generate GPU kd-tree..." << std::endl;
                        GpuSurface gpu_surface(points, num_points);

                        gpu_surface.setKn(m_kn);
                        gpu_surface.setKi(m_ki);
                        gpu_surface.setFlippoint(m_flipPoint[0], m_flipPoint[1], m_flipPoint[2]);

                        gpu_surface.calculateNormals();
                        gpu_surface.getNormals(normals);

                        p_loader_reduced->setNormalArray(normals, num_points);
                        gpu_surface.freeGPU();
#else
                        std::cout << timestamp << "ERROR: GPU Driver not installed" << std::endl;
                        surface->calculateSurfaceNormals();
#endif
                    }
                    else
                    {
                        surface->calculateSurfaceNormals();
                    }
                }

                auto ps_grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(
                        m_voxelSizes[h], surface, gridbb, true, m_extrude);

                ps_grid->setBB(gridbb);
                ps_grid->calcIndices();
                ps_grid->calcDistanceValues();

                std::stringstream ss2;
                ss2 << name_id << ".ser";
                ps_grid->saveCells(ss2.str());
                grid_files.push_back(ss2.str());
                partitionBoxesNew.push_back(partitionBoxes->at(i));
            }
            std::cout << lvr2::timestamp << "Skipped PartitionBoxes: " << partitionBoxesSkipped << std::endl;

            auto vmax = cbb.getMax();
            auto vmin = cbb.getMin();
            vmin.x -= m_voxelSizes[h] * 3;
            vmin.y -= m_voxelSizes[h] * 3;
            vmin.z -= m_voxelSizes[h] * 3;
            vmax.x += m_voxelSizes[h] * 3;
            vmax.y += m_voxelSizes[h] * 3;
            vmax.z += m_voxelSizes[h] * 3;
            cbb.expand(vmin);
            cbb.expand(vmax);

            auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(grid_files, partitionBoxesNew, cbb, m_voxelSizes[h]);

            auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(hg);

            lvr2::HalfEdgeMesh<Vec> mesh;

            reconstruction->getMesh(mesh);

            if (m_removeDanglingArtifacts)
            {
                cout << timestamp << "Removing dangling artifacts" << endl;
                removeDanglingCluster(mesh, static_cast<size_t>(m_removeDanglingArtifacts));
            }

            // Magic number from lvr1 `cleanContours`...
            cleanContours(mesh, m_cleanContours, 0.0001);

            // Fill small holes if requested
            if (m_fillHoles)
            {
                naiveFillSmallHoles(mesh, m_fillHoles, false);
            }

            // Calculate normals for vertices
            auto faceNormals = calcFaceNormals(mesh);

            ClusterBiMap<FaceHandle> clusterBiMap;
            if (m_optimizePlanes) {
                clusterBiMap = iterativePlanarClusterGrowing(mesh,
                                                             faceNormals,
                                                             m_planeNormalThreshold,
                                                             m_planeIterations,
                                                             m_minPlaneSize);

                if (m_smallRegionThreshold > 0) {
                    deleteSmallPlanarCluster(
                            mesh, clusterBiMap, static_cast<size_t>(m_smallRegionThreshold));
                }

                double end_s = lvr2::timestamp.getElapsedTimeInS();

                if (m_retesselate) {
                    Tesselator<Vec>::apply(
                            mesh, clusterBiMap, faceNormals, m_lineFusionThreshold);
                }
            } else {
                clusterBiMap = planarClusterGrowing(mesh, faceNormals, m_planeNormalThreshold);
            }

            stringstream largeScale;
            string voxelSize = std::to_string(m_voxelSizes[h]);
            std::replace( voxelSize.begin(), voxelSize.end(), '.', '_');
            largeScale << "largeScale_" << voxelSize <<".ply";

            // Finalize mesh
            lvr2::SimpleFinalizer<Vec> finalize;
            auto meshBuffer = finalize.apply(mesh);

            auto m = ModelPtr(new Model(meshBuffer));
            ModelFactory::saveModel(m, largeScale.str());
        }

        // Is the return value actually used somewhere???
        return 1;

    }

    template <typename BaseVecT>
    int LargeScaleReconstruction<BaseVecT>::mpiChunkAndReconstruct(
        ScanProjectEditMarkPtr project,
        BoundingBox<BaseVecT>& newChunksBB,
        std::shared_ptr<ChunkHashGrid> chunkManager)
    {
        unsigned long timeStart1 = lvr2::timestamp.getCurrentTimeInMs();
        unsigned long timeSum = 0;
        m_chunkSize = chunkManager->getChunkSize();

        if(project->project->positions.size() != project->changed.size())
        {
            cout << "Inconsistency between number of given scans and diff-vector (scans to consider)! exit..." << endl;
            return 0;
        }

        cout << lvr2::timestamp << "Starting BigGrid" << endl;
        BigGrid<BaseVecT> bg(m_bgVoxelSize, project, m_scale);
        cout << lvr2::timestamp << "BigGrid finished " << endl;

        BoundingBox<BaseVecT> bb = bg.getBB();

        std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes;
        vector<BoundingBox<BaseVecT>> partitionBoxesNew;
        BoundingBox<BaseVecT> cmBB = BoundingBox<BaseVecT>();




        BoundingBox<BaseVecT> partbb = bg.getpartialBB();
        cout << lvr2::timestamp << "Generating VGrid" << endl;

        VirtualGrid<BaseVecT> vGrid(bg.getpartialBB(), m_chunkSize, m_bgVoxelSize);
        vGrid.calculateBoxes();
        partitionBoxes = vGrid.getBoxes();
        BaseVecT addMin = BaseVecT(std::floor(partbb.getMin().x / m_chunkSize) * m_chunkSize, std::floor(partbb.getMin().y / m_chunkSize) * m_chunkSize, std::floor(partbb.getMin().z / m_chunkSize) * m_chunkSize);
        BaseVecT addMax = BaseVecT(std::ceil(partbb.getMax().x / m_chunkSize) * m_chunkSize, std::ceil(partbb.getMax().y / m_chunkSize) * m_chunkSize, std::ceil(partbb.getMax().z / m_chunkSize) * m_chunkSize);
        newChunksBB.expand(addMin);
        newChunksBB.expand(addMax);
        std::cout << lvr2::timestamp << "Ginished vGrid" << endl;
        std::cout << lvr2::timestamp << "got: " << partitionBoxes->size() << " chunks." << std::endl;

        // we use the BB of all scans (including old ones) they are already hashed in the cm
        // and we can't make the BB smaller
        BaseVecT addCMBBMin = BaseVecT(std::floor(bb.getMin().x / m_chunkSize) * m_chunkSize, std::floor(bb.getMin().y / m_chunkSize) * m_chunkSize, std::floor(bb.getMin().z / m_chunkSize) * m_chunkSize);
        BaseVecT addCMBBMax = BaseVecT(std::ceil(bb.getMax().x / m_chunkSize) * m_chunkSize, std::ceil(bb.getMax().y / m_chunkSize) * m_chunkSize, std::ceil(bb.getMax().z / m_chunkSize) * m_chunkSize);
        cmBB.expand(addCMBBMin);
        cmBB.expand(addCMBBMax);

        chunkManager->setBoundingBox(cmBB);
        int numChunks_global = (cmBB.getXSize() / m_chunkSize) * (cmBB.getYSize() / m_chunkSize) * (cmBB.getZSize() / m_chunkSize);
        int numChunks_partial = partitionBoxes->size();

        cout << lvr2::timestamp << "Saving " << numChunks_global - numChunks_partial << " Chunks compared to full reconstruction" << endl;

        BaseVecT bb_min(bb.getMin().x, bb.getMin().y, bb.getMin().z);
        BaseVecT bb_max(bb.getMax().x, bb.getMax().y, bb.getMax().z);
        BoundingBox<BaseVecT> cbb(bb_min, bb_max);


        uint partitionBoxesSkipped = 0;

        std::cout << lvr2::timestamp << "Number of Voxels: " << m_voxelSizes.size() << std::endl;

        for(int h = 0; h < m_voxelSizes.size(); h++)
        {
            float voxelSize = m_voxelSizes[h];
            float overlap = 5 * voxelSize;

            if (m_debugChunks && h == 0)
            {
                boost::filesystem::remove_all("chunks");
                boost::filesystem::create_directories("chunks");

                YAML::Node metadata;
                metadata["chunk_size"] = m_chunkSize;
                metadata["voxel_size"] = voxelSize;
                metadata["overlap"] = overlap;

                std::ofstream out("chunks/chunk_metadata.yaml");
                out << metadata;
            }

            // vector to save the new chunk names - which chunks have to be reconstructed
            vector<BaseVector<int>> newChunks = vector<BaseVector<int>>();

            string layerName = "tsdf_values_" + std::to_string(voxelSize);
            //create chunks

            for (int i = 0; i < partitionBoxes->size(); i++)
            {
                string name_id;
                name_id =
                        std::to_string(
                                    (int)floor(partitionBoxes->at(i).getCentroid().x / m_chunkSize)) +
                                    "_" +std::to_string(
                                    (int)floor(partitionBoxes->at(i).getCentroid().y / m_chunkSize)) +
                            "_" + std::to_string((int)floor(partitionBoxes->at(i).getCentroid().z / m_chunkSize));


                size_t numPoints;

                BaseVecT gridbb_min = partitionBoxes->at(i).getMin();
                BaseVecT gridbb_max = partitionBoxes->at(i).getMax();
                for (int j = 0; j < 3; j++)
                {
                    gridbb_min[j] -= overlap;
                    gridbb_max[j] += overlap;
                }

                floatArr points = bg.points(gridbb_min, gridbb_max, numPoints);

                // remove chunks with less than 50 points
                if (numPoints <= 50)
                {
                    partitionBoxesSkipped++;
                    continue;
                }

                BoundingBox<BaseVecT> gridbb(gridbb_min, gridbb_max);

                cout <<  lvr2::timestamp <<"Processed Partitions: " << i << "/" << partitionBoxes->size() - 1 << endl;

                lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);
                p_loader->setPointArray(points, numPoints);

                if (bg.hasNormals())
                {
                    size_t numNormals;
                    lvr2::floatArr normals = bg.normals(gridbb_min, gridbb_max, numNormals);

                    p_loader->setNormalArray(normals, numNormals);
                    cout << "got " << numNormals << " normals" << endl;
                }

                lvr2::PointBufferPtr p_loader_reduced;
                //if(numPoints > (m_chunkSize*500000)) // reduction TODO add options
                if(false)
                {
                    OctreeReduction oct(p_loader, voxelSize / 5, 20);
                    p_loader_reduced = oct.getReducedPoints();
                }
                else
                {
                    p_loader_reduced = p_loader;
                }

                lvr2::PointsetSurfacePtr<Vec> surface;
                surface = make_shared<lvr2::AdaptiveKSearchSurface<Vec>>(p_loader_reduced,
                                                                         "FLANN",
                                                                         m_kn,
                                                                         m_ki,
                                                                         m_kd,
                                                                         m_useRansac);


                auto ps_grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(voxelSize, surface, gridbb, true, m_extrude);
                ps_grid->setBB(gridbb);
                ps_grid->calcIndices();
                bool distances_calculated = false;

                //calculate important stuff for reconstruction
                if (!bg.hasNormals())
                {
                    if (m_useGPU)
                    {
    #ifdef GPU_FOUND
                    // std::vector<float> flipPoint = std::vector<float>{100, 100, 100};
                    size_t num_points = p_loader_reduced->numPoints();
                    floatArr points = p_loader_reduced->getPointArray();
                    floatArr normals = floatArr(new float[num_points * 3]);
                    std::cout << timestamp << "Generate GPU kd-tree..." << std::endl;
                    GpuSurface gpu_surface(points, num_points);

                    gpu_surface.setKn(m_kn);
                    gpu_surface.setKi(m_ki);
                    gpu_surface.setKd(m_kd);
                    gpu_surface.setFlippoint(m_flipPoint[0], m_flipPoint[1], m_flipPoint[2]);

                    gpu_surface.calculateNormals();
                    gpu_surface.getNormals(normals);

                    p_loader_reduced->setNormalArray(normals, num_points);

                    auto& query_points = ps_grid->getQueryPoints();

                    if(m_useGPUDistances)
                    {
                        std::cout << timestamp << "Computing signed distances in GPU with brute force kernel (kd = 5)." << std::endl;
                        std::cout << timestamp << "This might take a while...." << std::endl;
                        gpu_surface.distances(query_points, m_voxelSizes[h]);
                        distances_calculated = true;
                        std::cout << timestamp << "Done." << std::endl;
                    }

                    gpu_surface.freeGPU();

    #else
                        std::cout << timestamp << "ERROR: GPU Driver not installed" << std::endl;
                        surface->calculateSurfaceNormals();
    #endif
                    }
                    else
                    {
                        surface->calculateSurfaceNormals();
                    }
                }

                if (!distances_calculated)
                {
                    ps_grid->calcDistanceValues();
                }

                unsigned long timeStart = lvr2::timestamp.getCurrentTimeInMs();
                int x = (int)floor(partitionBoxes->at(i).getCentroid().x / m_chunkSize);
                int y = (int)floor(partitionBoxes->at(i).getCentroid().y / m_chunkSize);
                int z = (int)floor(partitionBoxes->at(i).getCentroid().z / m_chunkSize);


                addTSDFChunkManager(x, y, z, ps_grid, chunkManager, layerName);
                BaseVector<int> chunkCoordinates(x, y, z);
                // also save the grid coordinates of the chunk added to the ChunkManager
                newChunks.push_back(chunkCoordinates);
                // also save the "real" bounding box without overlap
                partitionBoxesNew.push_back(partitionBoxes->at(i));

                unsigned long timeEnd = lvr2::timestamp.getCurrentTimeInMs();

                timeSum += timeEnd - timeStart;

                // save the mesh of the chunk
                if(m_debugChunks && h == 0)
                {
                    auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(ps_grid);
                    lvr2::PMPMesh<Vec> mesh;
                    reconstruction->getMesh(mesh);
                    if(mesh.numFaces() > 0)
                    {
                        mesh.getSurfaceMesh().write("chunks/" + name_id + ".ply");
                    }
                }

            }
            std::cout << lvr2::timestamp << "Skipped PartitionBoxes: " << partitionBoxesSkipped << std::endl;

            cout << "ChunkManagerIO Time: " <<(double) (timeSum / 1000.0) << " s" << endl;
            cout << lvr2::timestamp << "finished" << endl;

            if(m_bigMesh && h == 0)
            {
                //combine chunks
                auto vmin = cbb.getMin() - BaseVecT(overlap, overlap, overlap);
                auto vmax = cbb.getMax() + BaseVecT(overlap, overlap, overlap);
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
                                  << ") in layer: " << "tsdf_values_" + std::to_string(m_voxelSizes[0]) << std::endl;
                    }
                }
                auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(tsdfChunks, partitionBoxesNew, cbb,
                                                                                   m_voxelSizes[0]);
                tsdfChunks.clear();
                auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(hg);

                lvr2::HalfEdgeMesh<Vec> mesh;

                reconstruction->getMesh(mesh);

                if (m_removeDanglingArtifacts) {
                    cout << timestamp << "Removing dangling artifacts" << endl;
                    removeDanglingCluster(mesh, static_cast<size_t>(m_removeDanglingArtifacts));
                }

                // Magic number from lvr1 `cleanContours`...
                cleanContours(mesh, m_cleanContours, 0.0001);

                // Fill small holes if requested
                if (m_fillHoles) {
                    naiveFillSmallHoles(mesh, m_fillHoles, false);
                }

                // Calculate normals for vertices
                auto faceNormals = calcFaceNormals(mesh);

                ClusterBiMap<FaceHandle> clusterBiMap;
                if (m_optimizePlanes) {
                    clusterBiMap = iterativePlanarClusterGrowing(mesh,
                                                                 faceNormals,
                                                                 m_planeNormalThreshold,
                                                                 m_planeIterations,
                                                                 m_minPlaneSize);

                    if (m_smallRegionThreshold > 0) {
                        deleteSmallPlanarCluster(
                                mesh, clusterBiMap, static_cast<size_t>(m_smallRegionThreshold));
                    }

                    double end_s = lvr2::timestamp.getElapsedTimeInS();

                    if (m_retesselate) {
                        Tesselator<Vec>::apply(
                                mesh, clusterBiMap, faceNormals, m_lineFusionThreshold);
                    }
                } else {
                    clusterBiMap = planarClusterGrowing(mesh, faceNormals, m_planeNormalThreshold);
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

        unsigned long timeEnd1 = lvr2::timestamp.getCurrentTimeInMs();

        unsigned long timeSum1 = timeEnd1 - timeStart1;

        cout << "Finished complete reconstruction in " << (double) (timeSum1/1000.0) << "s" << endl;
        return 1;
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
        int chunksize = m_chunkSize;

        std::vector<PointBufferPtr> tsdfChunks;
        std::vector<BoundingBox<BaseVecT>> partitionBoxesNew = std::vector<BoundingBox<BaseVecT>>();
        BoundingBox<BaseVecT> completeBB = BoundingBox<BaseVecT>();

        int xMin = (int) (newChunksBB.getMin().x / m_chunkSize);
        int yMin = (int) (newChunksBB.getMin().y / m_chunkSize);
        int zMin = (int) (newChunksBB.getMin().z / m_chunkSize);

        int xMax = (int) (newChunksBB.getMax().x / m_chunkSize);
        int yMax = (int) (newChunksBB.getMax().y / m_chunkSize);
        int zMax = (int) (newChunksBB.getMax().z / m_chunkSize);

        std::cout << "DEBUG: New Chunks from (" << xMin << ", " << yMin << ", " << zMin
                  << ") - to (" << xMax << ", " << yMax << ", " << zMax << ")." << std::endl;

        for (int i = xMin - 1; i <= xMax + 1; i++) {
            for (int j = yMin - 1; j <= yMax + 1; j++) {
                for (int k = zMin - 1; k <= zMax + 1; k++) {
                    boost::optional<shared_ptr<PointBuffer>> chunk = chunkHashGrid->getChunk<PointBufferPtr>(layerName,
                                                                                                             i, j, k);


                    if (chunk) {
                        BaseVecT min(i * chunksize, j * chunksize, k * chunksize);
                        BaseVecT max(i * chunksize + chunksize, j * chunksize + chunksize, k * chunksize + chunksize);

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

        if (m_removeDanglingArtifacts) {
            cout << timestamp << "Removing dangling artifacts" << endl;
            removeDanglingCluster(mesh, static_cast<size_t>(m_removeDanglingArtifacts));
        }

        if (m_fillHoles) {
            naiveFillSmallHoles(mesh, m_fillHoles, false);
        }


        if (m_debugChunks)
        {
            auto faceNormals = calcFaceNormals(mesh);
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

        m_chunkSize = chunkManager->getChunkSize();

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
        BigGrid<BaseVecT> bg( m_bgVoxelSize ,project, m_scale);
        cout << lvr2::timestamp << "BigGrid finished " << endl;

        BoundingBox<BaseVecT> bb = bg.getBB();

        std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes;
        BoundingBox<BaseVecT> cmBB = BoundingBox<BaseVecT>();




        BoundingBox<BaseVecT> partbb = bg.getpartialBB();
        cout << lvr2::timestamp << "generating VGrid" << endl;

        VirtualGrid<BaseVecT> vGrid(
                bg.getpartialBB(), m_chunkSize, m_bgVoxelSize);
        vGrid.calculateBoxes();
        partitionBoxes = vGrid.getBoxes();
        BaseVecT addMin = BaseVecT(std::floor(partbb.getMin().x / m_chunkSize) * m_chunkSize, std::floor(partbb.getMin().y / m_chunkSize) * m_chunkSize, std::floor(partbb.getMin().z / m_chunkSize) * m_chunkSize);
        BaseVecT addMax = BaseVecT(std::ceil(partbb.getMax().x / m_chunkSize) * m_chunkSize, std::ceil(partbb.getMax().y / m_chunkSize) * m_chunkSize, std::ceil(partbb.getMax().z / m_chunkSize) * m_chunkSize);
        newChunksBB.expand(addMin);
        newChunksBB.expand(addMax);
        cout << lvr2::timestamp << "finished vGrid" << endl;
        std::cout << lvr2::timestamp << "got: " << partitionBoxes->size() << " Chunks"
                  << std::endl;

        // we use the BB of all scans (including old ones) they are already hashed in the cm
        // and we can't make the BB smaller
        BaseVecT addCMBBMin = BaseVecT(std::floor(bb.getMin().x / m_chunkSize) * m_chunkSize, std::floor(bb.getMin().y / m_chunkSize) * m_chunkSize, std::floor(bb.getMin().z / m_chunkSize) * m_chunkSize);
        BaseVecT addCMBBMax = BaseVecT(std::ceil(bb.getMax().x / m_chunkSize) * m_chunkSize, std::ceil(bb.getMax().y / m_chunkSize) * m_chunkSize, std::ceil(bb.getMax().z / m_chunkSize) * m_chunkSize);
        cmBB.expand(addCMBBMin);
        cmBB.expand(addCMBBMax);

        // TODO: It breaks here

        chunkManager->setBoundingBox(cmBB);
        int numChunks_global = (cmBB.getXSize() / m_chunkSize) * (cmBB.getYSize() / m_chunkSize) * (cmBB.getZSize() / m_chunkSize);
        int numChunks_partial = partitionBoxes->size();

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
        uint* partitionBoxesSkipped = new uint[m_voxelSizes.size()]();
        for(int h = 0; h < m_voxelSizes.size(); h++)
        {
            // vector to save the new chunk names - which chunks have to be reconstructed

            // send chunks

            for (int i = 0; i < partitionBoxes->size(); i++)
            {
                std::cout << lvr2::timestamp << "Chunk " << i+1 << "/" << partitionBoxes->size() << std::endl;


                size_t numPoints;

                BaseVecT gridbb_min = partitionBoxes->at(i).getMin();
                BaseVecT gridbb_max = partitionBoxes->at(i).getMax();
                for (int j = 0; j < 3; j++)
                {
                    gridbb_min[j] -= m_voxelSizes[h] * 3;
                    gridbb_max[j] += m_voxelSizes[h] * 3;
                }

                floatArr points = bg.points(gridbb_min, gridbb_max, numPoints);

                // remove chunks with less than 50 points
                if (numPoints <= 50)
                {
                    partitionBoxesSkipped[h]++;
                    continue;
                }

                // Get Bounding Box
                float x_min = gridbb_min.x, y_min = gridbb_min.y, z_min = gridbb_min.z,
                        x_max = gridbb_max.x, y_max = gridbb_max.y, z_max = gridbb_max.z;


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
                    lvr2::floatArr normals = bg.normals(gridbb_min, gridbb_max, numNormals);
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
        for(int h = 0; h < m_voxelSizes.size(); h++)
        {
            // vector to save the new chunk names - which chunks have to be reconstructed
            vector<BaseVector<int>> newChunks = vector<BaseVector<int>>();

            string layerName = "tsdf_values_" + std::to_string(m_voxelSizes[h]);

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
                int x = (int) floor(partitionBoxes->at(chunk).getCentroid().x / m_chunkSize);
                int y = (int) floor(partitionBoxes->at(chunk).getCentroid().y / m_chunkSize);
                int z = (int) floor(partitionBoxes->at(chunk).getCentroid().z / m_chunkSize);


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

            if(m_bigMesh && h == 0)
            {
                //combine chunks
                auto vmax = cbb.getMax();
                auto vmin = cbb.getMin();
                vmin.x -= m_voxelSizes[h] *3;
                vmin.y -= m_voxelSizes[h] *3;
                vmin.z -= m_voxelSizes[h] *3;
                vmax.x += m_voxelSizes[h] *3;
                vmax.y += m_voxelSizes[h] *3;
                vmax.z += m_voxelSizes[h] *3;
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
                                  << ") in layer: " << "tsdf_values_" + std::to_string(m_voxelSizes[0]) << std::endl;
                    }
                }
                auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(tsdfChunks, partitionBoxesNew, cbb,
                                                                                   m_voxelSizes[0]);
                tsdfChunks.clear();
                auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(hg);

                lvr2::HalfEdgeMesh<Vec> mesh;

                reconstruction->getMesh(mesh);

                if (m_removeDanglingArtifacts) {
                    cout << timestamp << "Removing dangling artifacts" << endl;
                    removeDanglingCluster(mesh, static_cast<size_t>(m_removeDanglingArtifacts));
                }

                // Magic number from lvr1 `cleanContours`...
                cleanContours(mesh, m_cleanContours, 0.0001);

                // Fill small holes if requested
                if (m_fillHoles) {
                    naiveFillSmallHoles(mesh, m_fillHoles, false);
                }

                // Calculate normals for vertices
                auto faceNormals = calcFaceNormals(mesh);

                ClusterBiMap<FaceHandle> clusterBiMap;
                if (m_optimizePlanes) {
                    clusterBiMap = iterativePlanarClusterGrowing(mesh,
                                                                 faceNormals,
                                                                 m_planeNormalThreshold,
                                                                 m_planeIterations,
                                                                 m_minPlaneSize);

                    if (m_smallRegionThreshold > 0) {
                        deleteSmallPlanarCluster(
                                mesh, clusterBiMap, static_cast<size_t>(m_smallRegionThreshold));
                    }

                    double end_s = lvr2::timestamp.getElapsedTimeInS();

                    if (m_retesselate) {
                        Tesselator<Vec>::apply(
                                mesh, clusterBiMap, faceNormals, m_lineFusionThreshold);
                    }
                } else {
                    clusterBiMap = planarClusterGrowing(mesh, faceNormals, m_planeNormalThreshold);
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

            BaseVecT gridbb_min(x_min,
                                y_min,
                                z_min);
            BaseVecT gridbb_max(x_max,
                                y_max,
                                z_max);
            BoundingBox<BaseVecT> gridbb(gridbb_min, gridbb_max);


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

            //if(numPoints > (m_chunkSize*500000)) // reduction TODO add options
            /*if(false)
            {
                OctreeReduction oct(p_loader, m_voxelSizes[h], 20);
                p_loader_reduced = oct.getReducedPoints();
            }
            else
            {
                p_loader_reduced = p_loader;
            }*/

            lvr2::PointsetSurfacePtr<Vec> surface;
            surface = make_shared<lvr2::AdaptiveKSearchSurface<Vec>>(p_loader,
                                                                     "FLANN",
                                                                     m_kn,
                                                                     m_ki,
                                                                     m_kd,
                                                                     m_useRansac);

            //calculate important stuff for reconstruction
            if (calcNorm)
            {
                surface->calculateSurfaceNormals();
            }
            auto ps_grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(
                    m_voxelSizes[h], surface, gridbb, true, m_extrude);

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
