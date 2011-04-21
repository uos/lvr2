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
#include "Options.hpp"
#include "reconstruction/StannPointCloudManager.hpp"
#include "reconstruction/FastReconstruction.hpp"
#include "io/PLYIO.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/TriangleMesh.hpp"
#include "geometry/HalfEdgeMesh.hpp"

using namespace lssr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    Options options(argc, argv);

    // Create a point cloud manager
    StannPointCloudManager<Vertex<float>, Normal<float> > manager(options.getOutputFileName(),
                                          	  	   options.getKn(),
                                          	  	   options.getKi(),
                                          	  	   options.getKd());

    // Create an empty mesh
    //TriangleMesh<Vertex<float>, unsigned int > mesh;
    HalfEdgeMesh<Vertex<float>, Normal<float> > mesh;

    // Create a new reconstruction object
    FastReconstruction<Vertex<float>, Normal<float> > reconstruction(manager, options.getVoxelsize());
    reconstruction.getMesh(mesh);

    // Save triangle mesh
    mesh.finalize();
    //mesh.save("triangle_mesh.ply");

    cout << timestamp << "Program end." << endl;

	return 0;
}

