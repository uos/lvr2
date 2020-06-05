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
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/hdf5/ChannelIO.hpp"
#include "lvr2/io/hdf5/ArrayIO.hpp"
#include "lvr2/io/hdf5/VariantChannelIO.hpp"
#include "lvr2/io/hdf5/MeshIO.hpp"
#include "lvr2/reconstruction/BigGrid.hpp"
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


#include "LargeScaleReconstruction.hpp"


#if defined CUDA_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/cuda/CudaSurface.hpp"

    typedef lvr2::CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/opencl/ClSurface.hpp"
    typedef lvr2::ClSurface GpuSurface;
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
        std::cout << "Reconstruction Instance generated..." << std::endl;
    }

    template<typename BaseVecT>
    LargeScaleReconstruction<BaseVecT>::LargeScaleReconstruction( vector<float> voxelSizes, float bgVoxelSize,
                                                                 float scale, uint nodeSize,
                                                                 int partMethod, int ki, int kd, int kn, bool useRansac,
                                                                 std::vector<float> flipPoint,
                                                                 bool extrude, int removeDanglingArtifacts,
                                                                 int cleanContours, int fillHoles, bool optimizePlanes,
                                                                 float planeNormalThreshold, int planeIterations,
                                                                 int minPlaneSize, int smallRegionThreshold,
                                                                 bool retesselate, float lineFusionThreshold,
                                                                 bool bigMesh, bool debugChunks, bool useGPU)
            : m_voxelSizes(voxelSizes), m_bgVoxelSize(bgVoxelSize),
              m_scale(scale),m_nodeSize(nodeSize),
              m_partMethod(partMethod), m_ki(ki), m_kd(kd), m_kn(kn), m_useRansac(useRansac),
              m_flipPoint(flipPoint), m_extrude(extrude), m_removeDanglingArtifacts(removeDanglingArtifacts),
              m_cleanContours(cleanContours), m_fillHoles(fillHoles), m_optimizePlanes(optimizePlanes),
              m_planeNormalThreshold(planeNormalThreshold), m_planeIterations(planeIterations),
              m_minPlaneSize(minPlaneSize), m_smallRegionThreshold(smallRegionThreshold),
              m_retesselate(retesselate), m_lineFusionThreshold(lineFusionThreshold),m_bigMesh(bigMesh), m_debugChunks(debugChunks), m_useGPU(useGPU)
    {
        std::cout << "Reconstruction Instance generated..." << std::endl;
    }

    template<typename BaseVecT>
    LargeScaleReconstruction<BaseVecT>::LargeScaleReconstruction(LSROptions options)
            : LargeScaleReconstruction(options.voxelSizes, options.bgVoxelSize,
              options.scale,options.nodeSize,
              options.partMethod, options.ki, options.kd, options.kn, options.useRansac,
              options.getFlipPoint(), options.extrude, options.removeDanglingArtifacts,
              options.cleanContours, options.fillHoles, options.optimizePlanes,
              options.planeNormalThreshold, options.planeIterations,
              options.minPlaneSize, options.smallRegionThreshold,
              options.retesselate, options.lineFusionThreshold, options.bigMesh, options.debugChunks, options.useGPU)
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

        cout << lvr2::timestamp << "finished tree" << endl;
        std::cout << lvr2::timestamp << "got: " << partitionBoxes->size() << " leafs, saving leafs"
                  << std::endl;

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

                floatArr points = bg.points(partitionBoxes->at(i).getMin().x - m_voxelSizes[h] * 3,
                                            partitionBoxes->at(i).getMin().y - m_voxelSizes[h] * 3,
                                            partitionBoxes->at(i).getMin().z - m_voxelSizes[h] * 3,
                                            partitionBoxes->at(i).getMax().x + m_voxelSizes[h] * 3,
                                            partitionBoxes->at(i).getMax().y + m_voxelSizes[h] * 3,
                                            partitionBoxes->at(i).getMax().z + m_voxelSizes[h] * 3,
                                            numPoints);

                // remove boxes with less than 50 points
                if (numPoints <= 50)
                {
                    partitionBoxesSkipped++;
                    continue;
                }

                BaseVecT gridbb_min(partitionBoxes->at(i).getMin().x - m_voxelSizes[h] * 3,
                                    partitionBoxes->at(i).getMin().y - m_voxelSizes[h] * 3,
                                    partitionBoxes->at(i).getMin().z - m_voxelSizes[h] * 3);
                BaseVecT gridbb_max(partitionBoxes->at(i).getMax().x + m_voxelSizes[h] * 3,
                                    partitionBoxes->at(i).getMax().y + m_voxelSizes[h] * 3,
                                    partitionBoxes->at(i).getMax().z + m_voxelSizes[h] * 3);
                BoundingBox<BaseVecT> gridbb(gridbb_min, gridbb_max);

                cout << "\n" <<  lvr2::timestamp <<"box: " << i << "/" << partitionBoxes->size() - 1 << endl;

                lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);
                p_loader->setPointArray(points, numPoints);

                if (bg.hasNormals())
                {
                    size_t numNormals;
                    lvr2::floatArr normals = bg.normals(partitionBoxes->at(i).getMin().x - m_voxelSizes[h] * 3,
                                                        partitionBoxes->at(i).getMin().y - m_voxelSizes[h] * 3,
                                                        partitionBoxes->at(i).getMin().z - m_voxelSizes[h] * 3,
                                                        partitionBoxes->at(i).getMax().x + m_voxelSizes[h] * 3,
                                                        partitionBoxes->at(i).getMax().y + m_voxelSizes[h] * 3,
                                                        partitionBoxes->at(i).getMax().z + m_voxelSizes[h] * 3,
                                                        numNormals);

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
    int LargeScaleReconstruction<BaseVecT>::mpiChunkAndReconstruct(ScanProjectEditMarkPtr project,
            BoundingBox<BaseVecT>& newChunksBB,
            std::shared_ptr<ChunkHashGrid> chunkManager)
    {
        unsigned long timeSum = 0;
        m_chunkSize = chunkManager->getChunkSize();

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

        chunkManager->setBoundingBox(cmBB);
        int numChunks_global = (cmBB.getXSize() / m_chunkSize) * (cmBB.getYSize() / m_chunkSize) * (cmBB.getZSize() / m_chunkSize);
        int numChunks_partial = partitionBoxes->size();

        cout << lvr2::timestamp << "Saving " << numChunks_global - numChunks_partial << " Chunks compared to full reconstruction" << endl;

        BaseVecT bb_min(bb.getMin().x, bb.getMin().y, bb.getMin().z);
        BaseVecT bb_max(bb.getMax().x, bb.getMax().y, bb.getMax().z);
        BoundingBox<BaseVecT> cbb(bb_min, bb_max);


        uint partitionBoxesSkipped = 0;

        for(int h = 0; h < m_voxelSizes.size(); h++)
        {
            // vector to save the new chunk names - which chunks have to be reconstructed
            vector<BaseVector<int>> newChunks = vector<BaseVector<int>>();

            string layerName = "tsdf_values_" + std::to_string(m_voxelSizes[h]);
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

                floatArr points = bg.points(partitionBoxes->at(i).getMin().x - m_voxelSizes[h] *3,
                                            partitionBoxes->at(i).getMin().y - m_voxelSizes[h] *3,
                                            partitionBoxes->at(i).getMin().z - m_voxelSizes[h] *3,
                                            partitionBoxes->at(i).getMax().x + m_voxelSizes[h] *3,
                                            partitionBoxes->at(i).getMax().y + m_voxelSizes[h] *3,
                                            partitionBoxes->at(i).getMax().z + m_voxelSizes[h] *3,
                                            numPoints);

                // remove chunks with less than 50 points
                if (numPoints <= 50)
                {
                    partitionBoxesSkipped++;
                    continue;
                }

                BaseVecT gridbb_min(partitionBoxes->at(i).getMin().x - m_voxelSizes[h] *3,
                                    partitionBoxes->at(i).getMin().y - m_voxelSizes[h] *3,
                                    partitionBoxes->at(i).getMin().z - m_voxelSizes[h] *3);
                BaseVecT gridbb_max(partitionBoxes->at(i).getMax().x + m_voxelSizes[h] *3,
                                    partitionBoxes->at(i).getMax().y + m_voxelSizes[h] *3,
                                    partitionBoxes->at(i).getMax().z + m_voxelSizes[h] *3);
                BoundingBox<BaseVecT> gridbb(gridbb_min, gridbb_max);

                cout << "\n" <<  lvr2::timestamp <<"grid: " << i << "/" << partitionBoxes->size() - 1 << endl;

                lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);
                p_loader->setPointArray(points, numPoints);

                if (bg.hasNormals())
                {
                    size_t numNormals;
                    lvr2::floatArr normals = bg.normals(partitionBoxes->at(i).getMin().x -m_voxelSizes[h] *3,
                                                        partitionBoxes->at(i).getMin().y -m_voxelSizes[h] *3,
                                                        partitionBoxes->at(i).getMin().z -m_voxelSizes[h] *3,
                                                        partitionBoxes->at(i).getMax().x +m_voxelSizes[h] *3,
                                                        partitionBoxes->at(i).getMax().y +m_voxelSizes[h] *3,
                                                        partitionBoxes->at(i).getMax().z +m_voxelSizes[h] *3,
                                                        numNormals);

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
                    auto reconstruction =
                            make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(ps_grid);
                    lvr2::HalfEdgeMesh<Vec> mesh;
                    reconstruction->getMesh(mesh);
                    if(mesh.numVertices() > 0 && mesh.numFaces() > 0)
                    {
                        lvr2::SimpleFinalizer<Vec> finalize;
                        auto meshBuffer = MeshBufferPtr(finalize.apply(mesh));
                        auto m = ModelPtr(new Model(meshBuffer));
                        ModelFactory::saveModel(m, name_id + ".ply");
                    }
                }

            }
            std::cout << lvr2::timestamp << "Skipped PartitionBoxes: " << partitionBoxesSkipped << std::endl;

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
}
