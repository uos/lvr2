/*
 * Main.cpp
 *
 *  Created on: 16.05.2020
 *      Author: Benedikt Schumacher (bschumacher@uos.de)
 */

/// New includes, to be evaluated, which we actually need

#include "Options.hpp"

#include "lvr2/algorithm/CleanupAlgorithms.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/types/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/DMCReconstruction.hpp"
#include "lvr2/reconstruction/metrics/MSRMetric.hpp"
#include "lvr2/reconstruction/metrics/DMCStepMetric.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/metrics/OneSidedHausdorffMetric.hpp"
#include "lvr2/reconstruction/metrics/SymmetricHausdorffMetric.hpp"
#include "lvr2/config/lvropenmp.hpp"

using namespace lvr2;

using Vec = BaseVector<float>;

template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(const dmc_reconstruction::Options& options)
{
    // Create a point loader object
    ModelPtr model = ModelFactory::readModel(options.getInputFileName());

    // Parse loaded data
    if (!model)
    {
        std::cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << std::endl;
        return nullptr;
    }

    PointBufferPtr buffer = model->m_pointCloud;

    // Create a point cloud manager
    string pcm_name = options.getPCM();
    PointsetSurfacePtr<Vec> surface;

    // Create point set surface object
    if(pcm_name == "PCL")
    {
        std::cout << timestamp << "Using PCL as point cloud manager is not implemented yet!" << std::endl;
        panic_unimplemented("PCL as point cloud manager");
    }
    else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
    {
        
        int plane_fit_method = 0;
        
        if(options.useRansac())
        {
            plane_fit_method = 1;
        }

        // plane_fit_method
        // - 0: PCA
        // - 1: RANSAC
        // - 2: Iterative

        surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer,
            pcm_name,
            options.getKn(),
            options.getKi(),
            options.getKd(),
            plane_fit_method,
            options.getScanPoseFile()
        );
    }
    else
    {
        std::cout << timestamp << "Unable to create PointCloudManager." << std::endl;
        std::cout << timestamp << "Unknown option '" << pcm_name << "'." << std::endl;
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

int main(int argc, char** argv)
{

    dmc_reconstruction::Options options(argc, argv);

    options.printLogo();

    // if one of the needed parameters is missing,
    if (options.printUsage())
    {
        return EXIT_SUCCESS;
    }

    std::cout << options << std::endl;


    // =======================================================================
    // Set number of threads
    // =======================================================================
    OpenMPConfig::setNumThreads(options.getNumThreads());

    // Create a point cloud manager
    string pcm_name = options.getPCM();
    auto surface = loadPointCloud<Vec>(options);
    if (!surface)
    {
        std::cout << "Failed to create pointcloud. Exiting." << std::endl;
        return EXIT_FAILURE;
    }


    int delta = 1;
    HalfEdgeMesh<Vec> flatMesh;
    HalfEdgeMesh<Vec> deepMesh;

    DMCReconstruction<Vec, FastBox<Vec>> dmc(surface, surface->getBoundingBox(), true, options.getMaxLevel(), options.getMaxError());

    // creating and compairing two meshes
    dmc.getMesh(flatMesh, deepMesh, delta);

    DMCStepMetric *metric = new SymmetricHausdorffMetric();

    // get distance between the two meshes
    metric->get_distance(flatMesh, deepMesh);
    
    // Finalize mesh
    lvr2::SimpleFinalizer<Vec> finalize;
    auto meshBuffer = finalize.apply(flatMesh);

    auto m = ModelPtr(new Model(meshBuffer));
    ModelFactory::saveModel(m, "flat_mesh.ply");

    meshBuffer = finalize.apply(deepMesh);
    m = ModelPtr(new Model(meshBuffer));
    ModelFactory::saveModel(m, "deep_mesh.ply");

    std::cout << timestamp << "Finished reconstruction" << std::endl;

    return 0;
}
