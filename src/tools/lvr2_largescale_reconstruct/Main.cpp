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
    string filePath = options.getInputFileName()[0];
    float voxelsize = options.getVoxelsize();
    float scale = options.getScaling();
    cout << lvr2::timestamp << "Starting grid" << endl;
    BigGrid<BaseVecT> bg(filePath, voxelsize, scale);
    cout << lvr2::timestamp << "grid finished " << endl;
    BoundingBox<BaseVecT> bb = bg.getBB();
    cout << bb << endl;

    // lvr2::floatArr points = bg.getPointCloud(numPoints);

    cout << lvr2::timestamp << "making tree" << endl;
    BigGridKdTree<BaseVecT> gridKd(bg.getBB(), options.getNodeSize(), &bg, voxelsize);
    gridKd.insert(bg.pointSize(), bg.getBB().getCentroid());
    cout << lvr2::timestamp << "finished tree" << endl;

    std::cout << lvr2::timestamp << "got: " << gridKd.getLeafs().size() << " leafs, saving leafs"
              << std::endl;

    BaseVecT bb_min(bb.getMin().x, bb.getMin().y, bb.getMin().z);
    BaseVecT bb_max(bb.getMax().x, bb.getMax().y, bb.getMax().z);
    BoundingBox<BaseVecT> cbb(bb_min, bb_max);

    vector<string> grid_files;

    for (int i = 0; i < gridKd.getLeafs().size(); i++)
    {
        size_t numPoints;

        // todo: okay?
        floatArr points = bg.points(gridKd.getLeafs()[i]->getBB().getMin().x,
                                    gridKd.getLeafs()[i]->getBB().getMin().y,
                                    gridKd.getLeafs()[i]->getBB().getMin().z,
                                    gridKd.getLeafs()[i]->getBB().getMax().x,
                                    gridKd.getLeafs()[i]->getBB().getMax().y,
                                    gridKd.getLeafs()[i]->getBB().getMax().z,
                                    numPoints);

        if (numPoints <= 50)
            continue;

        BaseVecT gridbb_min(gridKd.getLeafs()[i]->getBB().getMin().x,
                            gridKd.getLeafs()[i]->getBB().getMin().y,
                            gridKd.getLeafs()[i]->getBB().getMin().z);
        BaseVecT gridbb_max(gridKd.getLeafs()[i]->getBB().getMax().x,
                            gridKd.getLeafs()[i]->getBB().getMax().y,
                            gridKd.getLeafs()[i]->getBB().getMax().z);
        BoundingBox<BaseVecT> gridbb(gridbb_min, gridbb_max);

        cout << "grid: " << i << "/" << gridKd.getLeafs().size() - 1 << endl;
        cout << "grid has " << numPoints << " points" << endl;
        cout << "kn=" << options.getKn() << endl;
        cout << "ki=" << options.getKi() << endl;
        cout << "kd=" << options.getKd() << endl;
        cout << gridbb << endl;

        lvr2::PointBufferPtr p_loader(new lvr2::PointBuffer);
        p_loader->setPointArray(points, numPoints);

        if (bg.hasNormals())
        {
            size_t numNormals;
            lvr2::floatArr normals = bg.normals(gridKd.getLeafs()[i]->getBB().getMin().x,
                                                gridKd.getLeafs()[i]->getBB().getMin().y,
                                                gridKd.getLeafs()[i]->getBB().getMin().z,
                                                gridKd.getLeafs()[i]->getBB().getMax().x,
                                                gridKd.getLeafs()[i]->getBB().getMax().y,
                                                gridKd.getLeafs()[i]->getBB().getMax().z,
                                                numNormals);

            p_loader->setNormalArray(normals, numNormals);
            cout << "got " << numNormals << " normals" << endl;
        }

        lvr2::PointsetSurfacePtr<Vec> surface;
        surface = make_shared<lvr2::AdaptiveKSearchSurface<Vec>>(p_loader,
                                                                 "FLANN",
                                                                 options.getKn(),
                                                                 options.getKi(),
                                                                 options.getKd(),
                                                                 options.useRansac());

        if (!bg.hasNormals())
            surface->calculateSurfaceNormals();

        auto ps_grid = std::make_shared<lvr2::PointsetGrid<Vec, lvr2::FastBox<Vec>>>(
            voxelsize, surface, gridbb, true, options.extrude());

        ps_grid->setBB(gridbb);
        ps_grid->calcIndices();
        ps_grid->calcDistanceValues();

        auto reconstruction =
            make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(ps_grid);

        std::stringstream ss2;
        ss2 << "testgrid-" << i << ".ser";
        ps_grid->saveCells(ss2.str());
        grid_files.push_back(ss2.str());
    }

    cout << lvr2::timestamp << "finished" << endl;

    auto vmax = cbb.getMax();
    auto vmin = cbb.getMin();
    vmin.x -= voxelsize * 2;
    vmin.y -= voxelsize * 2;
    vmin.z -= voxelsize * 2;
    vmax.x += voxelsize * 2;
    vmax.y += voxelsize * 2;
    vmax.z += voxelsize * 2;
    cbb.expand(vmin);
    cbb.expand(vmax);

    auto hg = std::make_shared<HashGrid<BaseVecT, lvr2::FastBox<Vec>>>(grid_files, cbb, voxelsize);

    auto reconstruction = make_unique<lvr2::FastReconstruction<Vec, lvr2::FastBox<Vec>>>(hg);

    lvr2::HalfEdgeMesh<Vec> mesh;

    reconstruction->getMesh(mesh);

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

    /*mesh.setClassifier(options.getClassifier());
    mesh.getClassifier().setMinRegionSize(options.getSmallRegionThreshold());
    if (options.optimizePlanes())
    {
        mesh.optimizePlanes(options.getPlaneIterations(),
                            options.getNormalThreshold(),
                            options.getMinPlaneSize(),
                            options.getSmallRegionThreshold(),
                            true);

        mesh.fillHoles(options.getFillHoles());
        mesh.optimizePlaneIntersections();
        mesh.restorePlanes(options.getMinPlaneSize());

        if (options.getNumEdgeCollapses())
        {
            QuadricVertexCosts<ColorVertex<float, unsigned char>, Normal<float>> c =
                QuadricVertexCosts<ColorVertex<float, unsigned char>, Normal<float>>(true);
            mesh.reduceMeshByCollapse(options.getNumEdgeCollapses(), c);
        }
    }
    else if (options.clusterPlanes())
    {
        mesh.clusterRegions(options.getNormalThreshold(), options.getMinPlaneSize());
        mesh.fillHoles(options.getFillHoles());
    }

    if (options.retesselate())
    {
        mesh.finalizeAndRetesselate(options.generateTextures(), options.getLineFusionThreshold());
    }
    else
    {
        mesh.finalize();
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
    auto meshBuffer = finalize.apply(mesh);

    // save mesh
    auto m = ModelPtr(new Model(meshBuffer));
    ModelFactory::saveModel(m, "largeScale.ply");

    return 0;
}

int main(int argc, char** argv)
{
    LargeScaleOptions::Options options(argc, argv);

    int i = mpiReconstruct<Vec>(options);

    return 0;
}
