/**
 * @mainpage LSSR Toolkit Documentation
 *
 * @section Introduction
 *
 * My intrduction
 *
 * @section LSSR Software Usage
 *
 * How to use the program
 *
 * @section Tutorials
 *
 * Some tutorials
 *
 */

#include "StannPointCloudManager.hpp"
#include "FastReconstruction.hpp"
#include "PLYIO.hpp"
#include "Options.hpp"
#include "Matrix4.hpp"
#include "TriangleMesh.hpp"

using namespace lssr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    Options options(argc, argv);

    // Print set parameters
    cout << options << endl;

    // Create a point cloud manager
    StannPointCloudManager<float> manager(options.getOutputFileName(),
                                          options.getKn(),
                                          options.getKi(),
                                          options.getKd());

    // Create an empty mesh
    TriangleMesh<Vertex<float>, size_t > mesh;

    // Create a new reconstruction object
    FastReconstruction<float, size_t> reconstruction(manager, options.getVoxelsize());
    reconstruction.getMesh(mesh);

	return 0;
}

