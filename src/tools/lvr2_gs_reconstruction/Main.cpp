/*
 * MainGS.cpp
 *
 *  Created on: somewhen.02.2019
 *      Author: Patrick Hoffmann (pahoffmann@uos.de)
 */

/// New includes, to be evaluated, which we actually need

#include "OptionsGS.hpp"
#include "lvr2/algorithm/CleanupAlgorithms.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/gs2/GrowingCellStructure.hpp"

#include <signal.h>

using namespace lvr2;

using Vec = BaseVector<float>;
HalfEdgeMesh<Vec> mesh;

template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(const gs_reconstruction::Options& options,
                                            PointBufferPtr buffer)
{
    // Create a point cloud manager
    string pcm_name = options.getPcm();
    PointsetSurfacePtr<BaseVecT> surface;

    // Create point set surface object
    if (pcm_name == "PCL")
    {
        cout << timestamp << "Using PCL as point cloud manager is not implemented yet!" << endl;
        panic_unimplemented("PCL as point cloud manager");
    }
    else if (pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" ||
             pcm_name == "NANOFLANN")
    {
        surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer, pcm_name, options.getKn(), options.getKi(), options.getKd(), 1, "");
    }
    else
    {
        cout << timestamp << "Unable to create PointCloudManager." << endl;
        cout << timestamp << "Unknown option '" << pcm_name << "'." << endl;
        return nullptr;
    }

    // Set search options for normal estimation and distance evaluation
    surface->setKd(options.getKd());
    surface->setKi(options.getKi());
    surface->setKn(options.getKn());

    // calc normals if there are none
    if (!buffer->hasNormals() && buffer.get()->numPoints() < 1000000)
    {
        surface->calculateSurfaceNormals();
    }

    return surface;
}

void saveMesh(int s = 0)
{
    if (s != 0)
    {
        std::cout << endl << "Received signal bit..." << endl;
    }
    SimpleFinalizer<Vec> fin;
    MeshBufferPtr res = fin.apply(mesh);

    ModelPtr m(new Model(res));

    cout << timestamp << "Saving mesh." << endl;
    ModelFactory::saveModel(m, "triangle_init_mesh.ply");
    exit(0);
}

int main(int argc, char** argv)
{

    gs_reconstruction::Options options(argc, argv);

    // if one of the needed parameters is missing,
    if (options.printUsage())
    {
        return EXIT_SUCCESS;
    }

    std::cout << options << std::endl;

    // try to parse the model
    ModelPtr model = ModelFactory::readModel(options.getInputFileName());

    // did model parse succeed
    if (!model)
    {
        cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
        return EXIT_FAILURE;
    }

    /* Catch ctr+c and save the Mesh.. */
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = saveMesh;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    PointBufferPtr buffer = model->m_pointCloud;

    if (!buffer)
    {
        cout << "Failed to create Buffer...exiting..." << endl;
        PointBuffer* pointBuffer = new PointBuffer(model.get()->m_mesh.get()->getVertices(),
                                                   model.get()->m_mesh.get()->numVertices());
        PointBufferPtr pointer(pointBuffer);
        buffer = pointer;
    }

    // Create a point cloud manager
    string pcm_name = options.getPcm();
    auto surface = loadPointCloud<Vec>(options, buffer);
    if (!surface)
    {
        cout << "Failed to create pointcloud. Exiting." << endl;
        return EXIT_FAILURE;
    }

    GrowingCellStructure<Vec, Normal<float>> gcs(surface);

    // set gcs variables
    gcs.setRuntime(options.getRuntime());
    gcs.setBasicSteps(options.getBasicSteps());
    gcs.setBoxFactor(options.getBoxFactor());
    gcs.setAllowMiss(options.getAllowMiss());
    gcs.setCollapseThreshold(options.getCollapseThreshold());
    gcs.setDecreaseFactor(options.getDecreaseFactor());
    gcs.setDeleteLongEdgesFactor(options.getDeleteLongEdgesFactor());
    gcs.setFilterChain(options.isFilterChain());
    gcs.setLearningRate(options.getLearningRate());
    gcs.setNeighborLearningRate(options.getNeighborLearningRate());
    gcs.setNumSplits(options.getNumSplits());
    gcs.setWithCollapse(options.getWithCollapse());
    gcs.setInterior(options.isInterior());
    gcs.setNumBalances(options.getNumBalances());

    gcs.getMesh(mesh);

    saveMesh();

    return 0;
}
