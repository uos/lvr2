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
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/DMCReconstruction.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"

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
        cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
        return nullptr;
    }

    PointBufferPtr buffer = model->m_pointCloud;

    // Create a point cloud manager
    string pcm_name = options.getPCM();
    PointsetSurfacePtr<Vec> surface;

    // Create point set surface object
    if(pcm_name == "PCL")
    {
        cout << timestamp << "Using PCL as point cloud manager is not implemented yet!" << endl;
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
        cout << "Failed to create pointcloud. Exiting." << endl;
        return EXIT_FAILURE;
    }

    HalfEdgeMesh<Vec> mesh;

    DMCReconstruction<Vec, FastBox<Vec>> dmc(surface, surface->getBoundingBox(), true, options.getMaxLevel(), options.getMaxError());

    dmc.getMesh(mesh);

    // Finalize mesh
    lvr2::SimpleFinalizer<Vec> finalize;
    auto meshBuffer = finalize.apply(mesh);

    auto m = ModelPtr(new Model(meshBuffer));
    ModelFactory::saveModel(m, "triangle_mesh.ply");

    cout << timestamp << "Finished reconstruction" << endl;

    return 0;
}
