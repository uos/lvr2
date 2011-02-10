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

#include "IndexedTriangleMesh.hpp"
#include "StannPointCloudManager.hpp"
#include "PLYIO.hpp"
#include "Options.hpp"
#include "Matrix4.hpp"

using namespace lssr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    Options options(argc, argv);

    // Read given input file
    PLYIO plyio;
    plyio.read(options.getOutputFileName());
    size_t numPoints;
    float** points = plyio.getIndexedVertexArray(numPoints);


    // Create a point cloud manager
    StannPointCloudManager<float> manager(points, 0, numPoints, Vertexf());

    // Save points and normals
    manager.save("normals.nor");
    manager.save("points.pts");

	return 0;
}

