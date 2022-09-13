#include <boost/filesystem.hpp>
#include <iostream>
#include <bitset>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "Options.hpp"

#include "CudaNormals.cuh"

using namespace lvr2;

// **********************************************************************************

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z)
{
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}
// ***********************************************************************************

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
    normals_raw[0] = 10;

    initNormals2(normals_raw, num_points);

    std::cout << normals_raw[0] << ", " << normals_raw[1] << ", " << normals_raw[2] << std::endl;
    std::cout << normals_raw[3 * 289251 + 0] << ", " << normals_raw[3 * 289251 + 1] << ", " << normals_raw[3 * 289251 + 2] << std::endl;


    pbuffer->setNormalArray(normals, num_points);

    ModelFactory::saveModel(model, "test.ply");

    free(normals_raw);

    int num = 1;

    float x = 0.67;
    float y = 0.34;
    float z = 0.12;

    unsigned int morton = morton3D(x, y, z);

    std::cout << std::bitset<64>(morton) << std::endl;

    return 0;
}
