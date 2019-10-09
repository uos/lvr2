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

#include "LargeScaleOptions.hpp"
#include "lvr2/algorithm/CleanupAlgorithms.hpp"
#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include "lvr2/algorithm/ReductionAlgorithms.hpp"
#include "lvr2/algorithm/Tesselator.hpp"
#include "lvr2/algorithm/UtilAlgorithms.hpp"
#include "lvr2/io/DataStruct.hpp"
#include "lvr2/io/LineReader.hpp"
#include "lvr2/io/Model.hpp"
#include "lvr2/io/PLYIO.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/reconstruction/BigGrid.hpp"
#include "lvr2/reconstruction/BigGridKdTree.hpp"
#include "lvr2/reconstruction/BigVolumen.hpp"
#include "lvr2/reconstruction/QueryPoint.hpp"
#include "lvr2/reconstruction/VirtualGrid.hpp"

#include <boost/algorithm/string/replace.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <flann/flann.hpp>
#include <fstream>
#include <iostream>
#include <lvr2/algorithm/FinalizeAlgorithms.hpp>
#include <lvr2/algorithm/NormalAlgorithms.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/geometry/ColorVertex.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/io/Model.hpp>
#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/io/Timestamp.hpp>
#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr2/reconstruction/FastBox.hpp>
#include <lvr2/reconstruction/FastReconstruction.hpp>
#include <lvr2/reconstruction/HashGrid.hpp>
#include <lvr2/reconstruction/PointsetGrid.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/reconstruction/SearchTreeFlann.hpp>
#include <random>
#include <sstream>
#include <string>

using std::cout;
using std::endl;
using namespace lvr2;

#if defined CUDA_FOUND
#define GPU_FOUND

#include <lvr2/reconstruction/cuda/CudaSurface.hpp>

typedef CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
#define GPU_FOUND

#include <lvr2/reconstruction/opencl/ClSurface.hpp>
typedef ClSurface GpuSurface;
#endif

struct duplicateVertex
{
    float x;
    float y;
    float z;
    unsigned int id;
};

using Vec = lvr2::BaseVector<float>;

typedef lvr2::PointsetSurface<lvr2::ColorVertex<float, unsigned char>> psSurface;
typedef lvr2::AdaptiveKSearchSurface<Vec> akSurface;

template <typename BaseVecT>
int mpiReconstruct(const LargeScaleOptions::Options& options)
{
    double datastruct_time = 0;
    double normal_time = 0;
    double dist_time = 0;
    double mesh_time = 0;
    double merge_time = 0;
    double dup_time = 0;

    double seconds = 0;
    size_t bufferSize = options.getLineReaderBuffer();

    std::vector<string> inputFiles = options.getInputFileName();
    std::string firstPath = options.getInputFileName()[0];
    bool gotSerializedBG = false;

    float voxelsize = options.getVoxelsize();
    float scale = options.getScaling();
    std::vector<float> flipPoint = options.getFlippoint();
    cout << lvr2::timestamp << "Starting grid" << endl;
    float volumenSize = (float)(options.getVolumenSize()); // 10 x 10 x 10 voxel
    std::shared_ptr<BigGrid<BaseVecT>> global_bg;

    std::shared_ptr<BigVolumen<BaseVecT>> bv;

    std::shared_ptr<BoundingBox<BaseVecT>> bb; // Bounding Box used for partial reconstruction
    if (firstPath.find(".ls") != std::string::npos)
    {
        gotSerializedBG = true;
        volumenSize = 0;
    }
    if (volumenSize <= 0)
    {
        double start_ss = lvr2::timestamp.getElapsedTimeInS();
        if (gotSerializedBG)
        {
            global_bg = std::make_shared<BigGrid<BaseVecT>>(firstPath);
        }
        else
        {
            global_bg =
                std::make_shared<BigGrid<BaseVecT>>(inputFiles, voxelsize, scale, bufferSize);

            if (!(options.getPartialReconstruct() == "NONE"))
            {
                bb = std::make_shared<BoundingBox<BaseVecT>>(options.getPartialReconstruct());

                std::cout << "Bounding BoX (by BB): " << bb << std::endl;
            }

            global_bg->serialize("serinfo.ls");
        }
        double end_ss = lvr2::timestamp.getElapsedTimeInS();
        seconds += (end_ss - start_ss);
        cout << lvr2::timestamp << "grid finished in" << (end_ss - start_ss) << "sec." << endl;
        // bb = global_bg->getBB();
        // cout << bb << endl;
        double end_ss2 = lvr2::timestamp.getElapsedTimeInS();
        datastruct_time = (end_ss2 - start_ss);
    }

    vector<BoundingBox<BaseVecT>> partitionBoxes;
    // lvr2::floatArr points = global_bg->getPointCloud(numPoints);
    cout << lvr2::timestamp << "Making tree" << endl;
    std::unordered_map<size_t, typename BigVolumen<BaseVecT>::VolumeCellInfo>* cells;
    std::vector<typename BigVolumen<BaseVecT>::VolumeCellInfo*> cell_vec;
    if (volumenSize > 0)
    {
        if (fmod(volumenSize, voxelsize) > 0.00001)
        {
            cerr << "ERROR: Size of Volume must be multiple of voxelsize e.g. Volume 12 and "
                    "voxelsize 2"
                 << endl;
            exit(-1);
        }
        cout << lvr2::timestamp << " getting BoundingBox" << endl;

        LineReader lr(inputFiles);
        bv = std::shared_ptr<BigVolumen<BaseVecT>>(
            new BigVolumen<BaseVecT>(inputFiles, volumenSize, volumenSize / 10));
        cells = bv->getCellinfo();
        for (auto cell_it = cells->begin(); cell_it != cells->end(); cell_it++)
        {
            cell_vec.push_back(&cell_it->second);
            BoundingBox<BaseVecT> partBB = cell_it->second.bb;
            partitionBoxes.push_back(partBB);
        }
    }
    else
    {
        if (gotSerializedBG)
        {
            ifstream partBoxIFS("KdTree.ser");
            while (partBoxIFS.good())
            {
                float minx, miny, minz, maxx, maxy, maxz;
                partBoxIFS >> minx;
                partBoxIFS >> miny;
                partBoxIFS >> minz;
                partBoxIFS >> maxx;
                partBoxIFS >> maxy;
                partBoxIFS >> maxz;
                BaseVecT bb_min(minx, miny, minz);
                BaseVecT bb_max(maxx, maxy, maxz);
                BoundingBox<BaseVecT> partBB(bb_min, bb_max);
                partitionBoxes.push_back(partBB);
            }
        }
        else
        {

            if (options.getVGrid() == 1)
            {
                VirtualGrid<BaseVecT> a(
                    global_bg->getBB(), options.getNodeSize(), options.getGridSize(), voxelsize);
                std::vector<shared_ptr<BoundingBox<BaseVecT>>> boxes;
                if (!(options.getPartialReconstruct() == "NONE"))
                {
                    a.setBoundingBox(*bb);
                }

                a.calculateBoxes();

                ofstream partBoxOfs("BoundingBoxes.ser");
                for (size_t i = 0; i < a.getBoxes().size(); i++)
                {
                    BoundingBox<BaseVecT> partBB = *a.getBoxes().at(i).get();
                    partitionBoxes.push_back(partBB);
                    partBoxOfs << partBB.getMin()[0] << " " << partBB.getMin()[1] << " "
                               << partBB.getMin()[2] << " " << partBB.getMax()[0] << " "
                               << partBB.getMax()[1] << " " << partBB.getMax()[2] << std::endl;
                }
            }
            else
            {
                BigGridKdTree<BaseVecT> gridKd(
                    global_bg->getBB(), options.getNodeSize(), global_bg.get(), voxelsize);
                gridKd.insert(global_bg->pointSize(), global_bg->getBB().getCentroid());
                ofstream partBoxOfs("KdTree.ser");
                for (size_t i = 0; i < gridKd.getLeafs().size(); i++)
                {
                    BoundingBox<BaseVecT> partBB = gridKd.getLeafs()[i]->getBB();
                    partitionBoxes.push_back(partBB);
                    partBoxOfs << partBB.getMin()[0] << " " << partBB.getMin()[1] << " "
                               << partBB.getMin()[2] << " " << partBB.getMax()[0] << " "
                               << partBB.getMax()[1] << " " << partBB.getMax()[2] << std::endl;
                }
            }
        }
    }

    cout << lvr2::timestamp << "finished tree" << endl;

    std::cout << lvr2::timestamp << "got: " << partitionBoxes.size() << " leafs, saving leafs"
              << std::endl;

    BaseVecT cbb_min = global_bg->getBB().getMin();
    BaseVecT cbb_max = global_bg->getBB().getMax();
    BoundingBox<BaseVecT> cbb(cbb_min, cbb_max);

    // vector contains the amount of vertices per grid
    vector<size_t> offsets;
    offsets.push_back(0);

    unordered_set<string> mesh_files;
    vector<string> grid_files;
    vector<string> normal_files;
    for (size_t i = 0; i < partitionBoxes.size(); i++)
    {
        string name_id;
        if (options.getVGrid() == 1)
        {
            name_id =
                std::to_string(
                    (int)floor(partitionBoxes.at(i).getMin().x / options.getGridSize())) +
                "_" +
                std::to_string(
                    (int)floor(partitionBoxes.at(i).getMin().y / options.getGridSize())) +
                "_" +
                std::to_string((int)floor(partitionBoxes.at(i).getMin().z / options.getGridSize()));
        }
        else
        {
            name_id = std::to_string(i);
        }
        size_t numPoints;

        // todo: okay?
        cout << lvr2::timestamp << "loading data " << i << endl;
        double start_s = lvr2::timestamp.getElapsedTimeInS();
        lvr2::floatArr points;
        lvr2::ucharArr colors;
        lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);

        if (volumenSize > 0)
        {

            stringstream ss_name;
            ss_name << "part-" << cell_vec[i]->ix << "-" << cell_vec[i]->iy << "-"
                    << cell_vec[i]->iz;
            name_id = ss_name.str();
            numPoints = cell_vec[i]->size + cell_vec[i]->overlapping_size;
            std::cout << ss_name.str() << ": " << cell_vec[i]->size << " + "
                      << cell_vec[i]->overlapping_size << " = " << numPoints << std::endl;
            if (numPoints < options.getKd() * 4 || numPoints < options.getKn() * 4 ||
                numPoints < options.getKi() * 4)
                continue;
            stringstream ss_points;
            ss_points << "part-" << cell_vec[i]->ix << "-" << cell_vec[i]->iy << "-"
                      << cell_vec[i]->iz << "-points.binary";
            cout << "file= " << ss_points.str() << endl;
            std::ifstream size_in(ss_points.str(), std::ifstream::ate | std::ifstream::binary);
            size_t file_bytes = size_in.tellg();
            cout << "OLD NUM: " << numPoints << endl;
            //            numPoints = (file_bytes/ sizeof(float))/3;
            points = lvr2::floatArr(new float[numPoints * 3]);
            cout << "NEW NUM: " << numPoints << endl;
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            // ifstream ifs_points(ss_points.str());
            ifs_points.read((char*)points.get(), sizeof(float) * 3 * numPoints);

            size_t readNum = numPoints;

            std::stringstream ss_normals2;
            ss_normals2 << "part-" << cell_vec[i]->ix << "-" << cell_vec[i]->iy << "-"
                        << cell_vec[i]->iz << "-points.ply";
            cout << "PART123: " << ss_normals2.str() << endl;
            PointBufferPtr pt(new PointBuffer);
            pt->setPointArray(points, numPoints);
            ModelPtr m(new Model(pt));
            ModelFactory::saveModel(m, ss_normals2.str());
            p_loader->setPointArray(points, numPoints);
        }
        else
        {
            std::cout << partitionBoxes[i] << std::endl;
            points = global_bg->points(partitionBoxes[i].getMin().x - voxelsize * 3,
                                       partitionBoxes[i].getMin().y - voxelsize * 3,
                                       partitionBoxes[i].getMin().z - voxelsize * 3,
                                       partitionBoxes[i].getMax().x + voxelsize * 3,
                                       partitionBoxes[i].getMax().y + voxelsize * 3,
                                       partitionBoxes[i].getMax().z + voxelsize * 3,
                                       numPoints);
            p_loader->setPointArray(points, numPoints);
            if (options.savePointNormals() || options.onlyNormals())
            {
                if (global_bg->hasColors())
                {
                    size_t numColors;
                    colors = global_bg->colors(partitionBoxes[i].getMin().x - voxelsize * 3,
                                               partitionBoxes[i].getMin().y - voxelsize * 3,
                                               partitionBoxes[i].getMin().z - voxelsize * 3,
                                               partitionBoxes[i].getMax().x + voxelsize * 3,
                                               partitionBoxes[i].getMax().y + voxelsize * 3,
                                               partitionBoxes[i].getMax().z + voxelsize * 3,
                                               numColors);
                    cout << "got ************* " << numColors << " colors" << endl;
                    p_loader->setColorArray(colors, numColors);
                }
            }
        }

        double end_s = lvr2::timestamp.getElapsedTimeInS();
        seconds += (end_s - start_s);
        cout << lvr2::timestamp << "finished loading data " << i << " in " << (end_s - start_s)
             << endl;
        // std::cout << "i: " << std::endl << bb << std::endl << "got : " << numPoints << std::endl;
        if (numPoints <= 50)
        {
            std::cout << "remove virtually empty box..." << std::endl;
            continue;
        }
        lvr2::BoundingBox<Vec> gridbb(Vec(partitionBoxes[i].getMin().x,
                                          partitionBoxes[i].getMin().y,
                                          partitionBoxes[i].getMin().z),
                                      Vec(partitionBoxes[i].getMax().x,
                                          partitionBoxes[i].getMax().y,
                                          partitionBoxes[i].getMax().z));

        cout << "grid: " << i << "/" << partitionBoxes.size() - 1 << endl;
        cout << "grid has " << numPoints << " points" << endl;
        cout << "kn=" << options.getKn() << endl;
        cout << "ki=" << options.getKi() << endl;
        cout << "kd=" << options.getKd() << endl;
        cout << gridbb << endl;

        bool navail = false;
        if (volumenSize <= 0)
        {
            if (global_bg->hasNormals())
            {

                size_t numNormals;
                lvr2::floatArr normals =
                    global_bg->normals(partitionBoxes[i].getMin().x - voxelsize * 3,
                                       partitionBoxes[i].getMin().y - voxelsize * 3,
                                       partitionBoxes[i].getMin().z - voxelsize * 3,
                                       partitionBoxes[i].getMax().x + voxelsize * 3,
                                       partitionBoxes[i].getMax().y + voxelsize * 3,
                                       partitionBoxes[i].getMax().z + voxelsize * 3,
                                       numNormals);

                p_loader->setNormalArray(normals, numNormals);
                navail = true;
            }
        }
        else
        {
            if (bv->hasNormals())
            {
                std::cout << "reading normals" << std::endl;
                size_t numNormals = numPoints;
                stringstream ss_normals;
                ss_normals << "part-" << cell_vec[i]->ix << "-" << cell_vec[i]->iy << "-"
                           << cell_vec[i]->iz << "-normals.binary";
                ifstream ifs_normals(ss_normals.str(), std::ifstream::binary);
                lvr2::floatArr normals(new float[numNormals * 3]);
                ifs_normals.read((char*)normals.get(), sizeof(float) * 3 * numNormals);
                p_loader->setNormalArray(normals, numNormals);
                if (bv->hasNormals())
                    navail = true;
            }
        }

        if (navail)
        {
        }
        else
        {
#ifdef GPU_FOUND
            if (options.useGPU())
            {
                std::cout << "calculating normals of " << numPoints << " points" << std::endl;
                if (numPoints > 30000000)
                    std::cout << "this is a lot of points, this might fail" << std::endl;
                double normal_start = lvr2::timestamp.getElapsedTimeInS();
                floatArr normals = floatArr(new float[numPoints * 3]);
                cout << timestamp << "Constructing kd-tree..." << endl;
                GpuSurface gpu_surface(points, numPoints);
                cout << timestamp << "Finished kd-tree construction." << endl;
                gpu_surface.setKn(options.getKn());
                gpu_surface.setKi(options.getKi());
                gpu_surface.setFlippoint(flipPoint[0], flipPoint[1], flipPoint[2]);
                cout << timestamp << "Start Normal Calculation..." << endl;
                gpu_surface.calculateNormals();
                gpu_surface.getNormals(normals);
                cout << timestamp << "Finished Normal Calculation. " << endl;
                p_loader->setNormalArray(normals, numPoints);
                gpu_surface.freeGPU();
                double normal_end = lvr2::timestamp.getElapsedTimeInS();
                normal_time += (normal_end - normal_start);
            }
#else
            cout << "ERROR: OpenCl not found" << endl;
            //                exit(-1);
#endif
        }

        // auto buffer = make_shared<lvr2::PointBuffer<Vec>>(*p_loader);
        lvr2::PointsetSurfacePtr<Vec> surface;
        surface = make_shared<lvr2::AdaptiveKSearchSurface<Vec>>(p_loader,
                                                                 "FLANN",
                                                                 options.getKn(),
                                                                 options.getKi(),
                                                                 options.getKd(),
                                                                 options.useRansac());

        if (navail)
        {
        }
        else if (!options.useGPU())
        {
            double normal_start = lvr2::timestamp.getElapsedTimeInS();
            surface->calculateSurfaceNormals();
            double normal_end = lvr2::timestamp.getElapsedTimeInS();
            normal_time += (normal_end - normal_start);
        }

        // Save points and normals only
        if (options.savePointNormals())
        {
            ModelPtr pn(new Model(surface->pointBuffer()));

            std::stringstream ss_normals;
            ss_normals << name_id << "-normals.ply";

            ModelFactory::saveModel(pn, ss_normals.str());

            normal_files.push_back(ss_normals.str());
        }

        if (options.onlyNormals())
            continue;
        double grid_start = lvr2::timestamp.getElapsedTimeInS();

        auto grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(
            voxelsize, surface, gridbb, true, options.extrude());
        grid->setBB(gridbb);
        grid->calcDistanceValues();
        auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(grid);

        double grid_end = lvr2::timestamp.getElapsedTimeInS();
        dist_time += (grid_end - grid_start);

        double mesh_start = lvr2::timestamp.getElapsedTimeInS();

        lvr2::HalfEdgeMesh<Vec> mesh;

        cout << lvr2::timestamp << " saving data " << i << endl;
        vector<unsigned int> duplicates;
        // reconstruction->getMesh(mesh);
        reconstruction->getMesh(mesh, grid->qp_bb, duplicates, voxelsize * 5);

        // =======================================================================
        // Optimize mesh
        // =======================================================================
        if (options.getDanglingArtifacts())
        {
            cout << timestamp << "Removing dangling artifacts" << endl;
            removeDanglingCluster(mesh, static_cast<size_t>(options.getDanglingArtifacts()));
        }

        // Magic number from lvr1 `cleanContours`...
        cleanContours(mesh, options.getCleanContourIterations(), 0.0001);

        // Fill small holes if requested
        if (options.getFillHoles())
        {
            naiveFillSmallHoles(mesh, options.getFillHoles(), false);
        }

        // Calculate normals for vertices
        auto faceNormals = calcFaceNormals(mesh);
        auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);

        // Reduce mesh complexity
        /*
        const auto reductionRatio = options.getEdgeCollapseReductionRatio();
        if (reductionRatio > 0.0)
        {
            if (reductionRatio > 1.0)
            {
                throw "The reduction ratio needs to be between 0 and 1!";
            }

            // Each edge collapse removes two faces in the general case.
            // TODO: maybe we should calculate this differently...
            const auto count = static_cast<size_t>((mesh.numFaces() / 2) * reductionRatio);
            auto collapsedCount = simpleMeshReduction(mesh, count, faceNormals);
        }*/

        ClusterBiMap<FaceHandle> clusterBiMap;
        if (options.optimizePlanes())
        {
            clusterBiMap = iterativePlanarClusterGrowing(mesh,
                                                         faceNormals,
                                                         options.getNormalThreshold(),
                                                         options.getPlaneIterations(),
                                                         options.getMinPlaneSize());

            if (options.getSmallRegionThreshold() > 0)
            {
                deleteSmallPlanarCluster(
                    mesh, clusterBiMap, static_cast<size_t>(options.getSmallRegionThreshold()));
            }

            if (options.retesselate())
            {
                Tesselator<Vec>::apply(
                    mesh, clusterBiMap, faceNormals, options.getLineFusionThreshold());
            }
        }
        else
        {
            clusterBiMap = planarClusterGrowing(mesh, faceNormals, options.getNormalThreshold());
        }

        // Finalize mesh
        lvr2::SimpleFinalizer<Vec> finalize;
        finalize.setNormalData(vertexNormals);
        auto meshBuffer = finalize.apply(mesh);

        double mesh_end = lvr2::timestamp.getElapsedTimeInS();
        mesh_time += mesh_end - mesh_start;
        start_s = lvr2::timestamp.getElapsedTimeInS();

        std::stringstream ss_mesh;

        ss_mesh << name_id << "-mesh.ply";
        mesh_files.insert(ss_mesh.str());

        // Create output model and save to file
        if (meshBuffer->numFaces() > 0)
        {
            auto m = ModelPtr(new Model(meshBuffer));

            ModelFactory::saveModel(m, ss_mesh.str());

            // add offset
            offsets.push_back(meshBuffer->numVertices() + offsets[offsets.size() - 1]);
        }

        std::stringstream ss_grid;
        ss_grid << name_id << "-grid.ser";
        //        ps_grid->saveCells(ss_grid.str());
        grid_files.push_back(ss_grid.str());

        /*
        std::stringstream ss_duplicates;
        ss_duplicates << name_id << "-duplicates.ser";
        std::ofstream ofs(ss_duplicates.str(), std::ofstream::out | std::ofstream::trunc);
        boost::archive::text_oarchive oa(ofs);
        oa& duplicates;
        */
        end_s = lvr2::timestamp.getElapsedTimeInS();
        seconds += (end_s - start_s);
        cout << lvr2::timestamp << "finished saving data " << i << " in " << (end_s - start_s)
             << endl;
    }

    vector<unsigned int> all_duplicates;
    //    vector<float> duplicateVertices;
    vector<duplicateVertex> duplicateVertices;
    /*for (int i = 0; i < grid_files.size(); i++)
    {
        string duplicate_path = grid_files[i];
        string ply_path = grid_files[i];
        double start_s = lvr2::timestamp.getElapsedTimeInS();
        boost::algorithm::replace_last(duplicate_path, "-grid.ser", "-duplicates.ser");
        boost::algorithm::replace_last(ply_path, "-grid.ser", "-mesh.ply");

        if (!(boost::filesystem::exists(ply_path)))
            continue;

        std::ifstream ifs(duplicate_path);
        boost::archive::text_iarchive ia(ifs);
        std::vector<unsigned int> duplicates;
        ia& duplicates;
        cout << "MESH " << i << " has " << duplicates.size() << " possible duplicates" << endl;

        lvr2::PLYIO io;
        ModelPtr modelPtr = io.read(ply_path);

        size_t numPoints = modelPtr->m_mesh->numVertices();

        offsets.push_back(numPoints + offsets[i]);
        if (numPoints == 0)
            continue;

        size_t numVertices = modelPtr->m_mesh->numVertices();
        floatArr modelVertices = modelPtr->m_mesh->getVertices();
        double end_s = lvr2::timestamp.getElapsedTimeInS();
        seconds += (end_s - start_s);

        for (size_t j = 0; j < duplicates.size(); j++)
        {
            duplicateVertex v;
            v.x = modelVertices[duplicates[j] * 3];
            v.y = modelVertices[duplicates[j] * 3 + 1];
            v.z = modelVertices[duplicates[j] * 3 + 2];
            v.id = duplicates[j] + offsets[i];
            duplicateVertices.push_back(v);
        }
    }*/

    /*std::sort(duplicateVertices.begin(),
              duplicateVertices.end(),
              [](const duplicateVertex& left, const duplicateVertex& right) {
                  return left.id < right.id;
              });*/
    //    std::sort(all_duplicates.begin(), all_duplicates.end());
    std::unordered_map<unsigned int, unsigned int> oldToNew;

    /*ofstream ofsd;
    ofsd.open("duplicate_colors.pts", ios_base::app);

    float comp_dist = std::max(voxelsize / 1000, 0.0001f);
    double dist_epsilon_squared = comp_dist * comp_dist;
    double dup_start = lvr2::timestamp.getElapsedTimeInS();

    string comment = lvr2::timestamp.getElapsedTime() + "Removing duplicates ";
    lvr2::ProgressBar progress(duplicateVertices.size(), comment);

    omp_lock_t writelock;
    omp_init_lock(&writelock);
#pragma omp parallel for
    for (size_t i = 0; i < duplicateVertices.size(); i++)
    {
        float ax = duplicateVertices[i].x;
        float ay = duplicateVertices[i].y;
        float az = duplicateVertices[i].z;
        //        vector<unsigned int> duplicates_of_i;
        int found = 0;
        auto find_it = oldToNew.find(duplicateVertices[i].id);
        if (find_it == oldToNew.end())
        {
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t j = 0; j < duplicateVertices.size(); j++)
            {
                if (i == j || found > 5)
                    continue;
                float bx = duplicateVertices[j].x;
                float by = duplicateVertices[j].y;
                float bz = duplicateVertices[j].z;
                double dist_squared =
                    (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz);
                if (dist_squared < dist_epsilon_squared)
                {
                    //

                    if (duplicateVertices[j].id < duplicateVertices[i].id)
                    {
                        //                        cout << "FUCK THIS SHIT" << endl;
                        continue;
                    }
                    omp_set_lock(&writelock);
                    oldToNew[duplicateVertices[j].id] = duplicateVertices[i].id;
                    found++;
                    omp_unset_lock(&writelock);
                }
            }
        }
        ++progress;
    }
    cout << "FOUND: " << oldToNew.size() << " duplicates" << endl;
    double dup_end = lvr2::timestamp.getElapsedTimeInS();
    dup_time += dup_end - dup_start;*/

    //    for(auto testit = oldToNew.begin(); testit != oldToNew.end(); testit++)
    //    {
    //        if(oldToNew.find(testit->second) != oldToNew.end()) cout << "SHIT FUCK SHIT" << endl;
    //    }

    ofstream ofs_vertices("largeVertices.bin", std::ofstream::out | std::ofstream::trunc);
    ofstream ofs_faces("largeFaces.bin", std::ofstream::out | std::ofstream::trunc);

    size_t increment = 0;
    std::map<size_t, size_t> decrements;
    decrements[0] = 0;

    size_t newNumVertices = 0;
    size_t newNumFaces = 0;

    /*############################################# Generating BigMesh here
     * ##############################################*/

    cout << lvr2::timestamp << "merging mesh..." << endl;

    size_t tmp_offset = 0;
    // TODO: add partial reconstruction here
    // TODO: filter out existing Meshes which overlap with new Meshes

    ifstream old_mesh("VGrid.ser");
    if (options.getVGrid() == 1 && old_mesh.is_open())
    {
        while (old_mesh.good())
        {
            string mesh;
            old_mesh >> mesh;
            mesh_files.insert(mesh);
        }
    }

    ofstream vGrid;
    vGrid.open("VGrid.ser", ofstream::out | ofstream::trunc);
    unordered_set<string>::iterator itr;

    bool vertexNormals = false;
    bool vertexColors = false;
    bool faceNormals = false;

    for (itr = mesh_files.begin(); itr != mesh_files.end(); itr++)
    {
        double start_s = lvr2::timestamp.getElapsedTimeInS();

        string ply_path = (*itr);
        // boost::algorithm::replace_last(ply_path, "-grid.ser", "-mesh.ply");

        if (!(boost::filesystem::exists(ply_path)))
            continue;

        vGrid << ply_path << std::endl;
        LineReader lr(ply_path);

        // size_t numPoints = lr.getNumPoints();
        // if (numPoints == 0)
        //    continue;

        lvr2::PLYIO io;
        ModelPtr modelPtr = io.read(ply_path);

        if (modelPtr->m_mesh->numVertices() == 0)
        {
            continue;
        }

        assert(modelPtr->m_mesh->numVertices() == lr.getNumPoints());

        size_t numVertices = modelPtr->m_mesh->numVertices();
        size_t numFaces = modelPtr->m_mesh->numFaces();

        // size_t offset = offsets[i];
        size_t offset = tmp_offset;
        floatArr modelVertices = modelPtr->m_mesh->getVertices();

        vertexNormals = modelPtr->m_mesh->hasVertexNormals();
        floatArr modelVertexNormals = modelPtr->m_mesh->getVertexNormals();

        vertexColors = modelPtr->m_mesh->hasVertexColors();
        size_t rgb = 3;
        ucharArr modelVertexColors = modelPtr->m_mesh->getVertexColors(rgb);

        uintArr modelFaces = modelPtr->m_mesh->getFaceIndices();

        faceNormals = modelPtr->m_mesh->hasFaceNormals();
        floatArr modelFaceNormals = modelPtr->m_mesh->getFaceNormals();

        double end_s = lvr2::timestamp.getElapsedTimeInS();
        seconds += (end_s - start_s);
        newNumVertices += numVertices;
        newNumFaces += numFaces;
        for (size_t j = 0; j < numVertices; j++)
        {
            float p[3];
            p[0] = modelVertices[j * 3];
            p[1] = modelVertices[j * 3 + 1];
            p[2] = modelVertices[j * 3 + 2];

            uchar pC[3];
            if (vertexColors)
            {
                pC[0] = modelVertexColors[j * 3];
                pC[1] = modelVertexColors[j * 3 + 1];
                pC[2] = modelVertexColors[j * 3 + 2];
            }

            float pN[3];
            if (vertexNormals)
            {
                pN[0] = modelVertexNormals[j * 3];
                pN[1] = modelVertexNormals[j * 3 + 1];
                pN[2] = modelVertexNormals[j * 3 + 2];
            }

            start_s = lvr2::timestamp.getElapsedTimeInS();
            ofs_vertices << std::setprecision(16) << p[0] << " " << p[1] << " " << p[2];

            if (vertexColors)
            {
                ofs_vertices << std::setprecision(16) << " " << pC[0] << " " << pC[1] << " "
                             << pC[2];
            }

            if (vertexNormals)
            {
                ofs_vertices << std::setprecision(16) << " " << pN[0] << " " << pN[1] << " "
                             << pN[2];
            }

            ofs_vertices << endl;

            end_s = lvr2::timestamp.getElapsedTimeInS();
            seconds += (end_s - start_s);
        }
        size_t new_face_num = 0;
        for (int j = 0; j < numFaces; j++)
        {
            unsigned int f[3];
            f[0] = modelFaces[j * 3] + offset;
            f[1] = modelFaces[j * 3 + 1] + offset;
            f[2] = modelFaces[j * 3 + 2] + offset;

            float fN[3];
            // termporarily disabled
            faceNormals = false;
            if (faceNormals)
            {
                fN[0] = modelFaceNormals[j * 3];
                fN[1] = modelFaceNormals[j * 3 + 1];
                fN[2] = modelFaceNormals[j * 3 + 2];
            }

            start_s = lvr2::timestamp.getElapsedTimeInS();
            ofs_faces << "3 " << f[0] << " " << f[1] << " " << f[2];

            if (faceNormals)
            {
                ofs_faces << std::setprecision(16) << " " << fN[0] << " " << fN[1] << " " << fN[2];
            }

            ofs_faces << endl;
            end_s = lvr2::timestamp.getElapsedTimeInS();
            seconds += (end_s - start_s);
        }
        tmp_offset += modelPtr->m_mesh->numVertices();
    }
    vGrid.close();
    ofs_faces.close();
    ofs_vertices.close();
    cout << lvr2::timestamp << "saving ply" << endl;
    cout << "Largest decrement: " << increment << endl;
    double start_s = lvr2::timestamp.getElapsedTimeInS();

    ofstream ofs_ply("bigMesh.ply", std::ofstream::out | std::ofstream::trunc);
    ifstream ifs_faces("largeFaces.bin");
    ifstream ifs_vertices("largeVertices.bin");
    string line;
    ofs_ply << "ply\n"
               "format ascii 1.0\n"
               "element vertex "
            << newNumVertices
            << "\n"
               "property float x\n"
               "property float y\n"
               "property float z\n";

    if (vertexColors)
    {
        ofs_ply << "property uchar red\n"
                   "property uchar green\n"
                   "property uchar blue\n";
    }

    if (vertexNormals)
    {
        ofs_ply << "property float nx\n"
                   "property float ny\n"
                   "property float nz\n";
    }

    ofs_ply << "element face " << newNumFaces
            << "\n"
               "property list uchar int vertex_indices\n"
               "end_header"
            << endl;

    while (std::getline(ifs_vertices, line))
    {
        ofs_ply << line << endl;
    }
    size_t c = 0;
    while (std::getline(ifs_faces, line))
    {

        ofs_ply << line << endl;
        stringstream ss(line);
        unsigned int v[3];
        ss >> v[0];
        ss >> v[1];
        ss >> v[2];
        for (int i = 0; i < 3; i++)
        {
            if (v[i] >= newNumVertices)
            {
                cout << "WTF: FACE " << c << " has index " << v[i] << endl;
            }
        }
        c++;
    }
    ofs_ply << endl;
    double end_s = lvr2::timestamp.getElapsedTimeInS();
    seconds += (end_s - start_s);

    cout << "IO-TIME: " << seconds << " seconds" << endl;
    cout << "DATASTRUCT-Time " << datastruct_time << endl;
    cout << "NORMAL-Time " << normal_time << endl;
    cout << "DIST-Time " << dist_time << endl;
    cout << "MESH-Time " << mesh_time << endl;
    cout << "MERGETime " << merge_time << endl;
    cout << "dup_time " << dup_time << endl;
    cout << lvr2::timestamp << "finished" << endl;
    cout << "saving largeNormal.ply" << endl;

    if (options.savePointNormals() || options.onlyNormals())
    {
        size_t numNormalPoints = 0;
        bool normalsWithColor = false;
        for (size_t i = 0; i < normal_files.size(); i++)
        {
            string ply_path = normal_files[i];
            LineReader lr(ply_path);
            size_t numPoints = lr.getNumPoints();
            numNormalPoints += numPoints;
            normalsWithColor = lr.getFileType() == XYZNRGB;
        }

        lvr2::floatArr normalPoints(new float[numNormalPoints * 3]);
        lvr2::floatArr normalNormals(new float[numNormalPoints * 3]);
        lvr2::ucharArr normalColors;
        if (normalsWithColor)
            normalColors = lvr2::ucharArr(new unsigned char[numNormalPoints * 3]);

        size_t globalId = 0;
        for (size_t i = 0; i < normal_files.size(); i++)
        {
            string ply_path = normal_files[i];

            auto m = ModelFactory::readModel(ply_path);
            if (m)
            {
                size_t amount = m->m_pointCloud->numPoints();
                auto p = m->m_pointCloud->getPointArray();
                if (p)
                {
                    for (size_t j = 0; j < amount * 3; j++)
                    {
                        normalPoints[globalId + j] = p[j];
                    }
                    size_t normalAmount = m->m_pointCloud->numPoints(); // not sure, if it's correct
                    auto n = m->m_pointCloud->getNormalArray();
                    for (size_t j = 0; j < normalAmount * 3; j++)
                    {
                        normalNormals[globalId + j] = n[j];
                    }
                    size_t colorAmount;
                    auto c = m->m_pointCloud->getColorArray(colorAmount);
                    for (size_t j = 0; j < colorAmount * 3; j++)
                    {
                        normalColors[globalId + j] = c[j];
                    }
                    globalId += amount * 3;
                }
            }
        }

        PointBufferPtr normalPB(new PointBuffer);
        normalPB->setPointArray(normalPoints, globalId / 3);
        normalPB->setNormalArray(normalNormals, globalId / 3);
        if (normalsWithColor)
        {
            normalPB->setColorArray(normalColors, globalId / 3);
        }
        ModelPtr nModel(new Model);
        nModel->m_pointCloud = normalPB;
        ModelFactory::saveModel(nModel, "bigNormals.ply");
    }

    ofs_ply.close();
    ifs_faces.close();
    ifs_vertices.close();

    return 0;
}

int main(int argc, char** argv)
{
    LargeScaleOptions::Options options(argc, argv);

    int i = mpiReconstruct<Vec>(options);

    return 0;
}
