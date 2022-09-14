#include <boost/filesystem.hpp>
#include <iostream>
#include <bitset>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "Options.hpp"

#include "CudaNormals.cuh"
#include "MortonCodes.cuh"

using namespace lvr2;

int main(int argc, char** argv)
{
    cuda_normals_2::Options opt(argc, argv);
    cout << opt << endl;

    // Get the model
    ModelPtr model = ModelFactory::readModel(opt.inputFile());

    // Get the points
    PointBufferPtr pbuffer = model->m_pointCloud;
    size_t num_points = model->m_pointCloud->numPoints();
    size_t size =  3 * num_points;

    floatArr points = pbuffer->getPointArray();
    
    float* points_raw = &points[0];

    // Create the normal array
    floatArr normals(new float[size]);

    float* normals_raw = (float*) malloc(sizeof(float) * 3 * num_points);

    // Initialize the normals
    initNormals2(normals_raw, num_points);

    // Set the normal array
    pbuffer->setNormalArray(normals, num_points);

    // Get the morton codes of the 3D points
    getMortonCodes(points_raw, num_points);

    // Save the new model as test.ply
    ModelFactory::saveModel(model, "test.ply");

    free(normals_raw);

    return 0;
}
