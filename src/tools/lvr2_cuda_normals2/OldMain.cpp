#include <boost/filesystem.hpp>
#include <iostream>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/util/IOUtils.hpp"
#include "Options.hpp"

using namespace lvr2;

int main(int argc, char** argv)
{
    cuda_normals_2::Options opt(argc, argv);
    cout << opt << endl;

    ModelPtr model = ModelFactory::readModel(opt.inputFile());

    PointBufferPtr pbuffer = model->m_pointCloud;
    size_t num_points = model->m_pointCloud->numPoints();

    floatArr points = pbuffer->getPointArray();
    
    float* points_raw = &points[0];
    
    floatArr normals(new float[num_points * 3]);

    for(size_t i = 0; i < num_points; i++)
    {
        normals[i * 3] = 1.0;
        normals[i * 3 + 1] = 0.0;
        normals[i * 3 + 2] = 1.0; 
    }

    pbuffer->setNormalArray(normals, num_points);

    ModelFactory::saveModel(model, "test.ply");

    return 0;
}