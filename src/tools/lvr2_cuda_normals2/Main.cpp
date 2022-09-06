#include <boost/filesystem.hpp>
#include <iostream>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "Options.hpp"

#include "CudaNormals.cuh"

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

    std::cout << num_points << std::endl;

    floatArr points = pbuffer->getPointArray();
    
    float* points_raw = &points[0];

    floatArr normals(new float[size]);

    float* normals_raw = (float*) malloc(sizeof(float) * 3 * num_points);

    //setNormals(argc, argv);
    initNormals(normals_raw, num_points);

    std::cout << normals_raw[0] << normals_raw[1] << normals_raw[2] << std::endl;

    pbuffer->setNormalArray(normals, num_points);

    ModelFactory::saveModel(model, "test.ply");

    free(normals_raw);

    return 0;
}
