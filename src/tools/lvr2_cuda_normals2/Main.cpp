#include <boost/filesystem.hpp>
#include <iostream>
#include <bitset>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "Options.hpp"

#include "kernels_host.h"

using namespace lvr2;

void floatToBinary(float f)
{
    char* bits = reinterpret_cast<char*>(&f);
    for(int n = sizeof(f) - 1; n >= 0; n--)
        std::cout << std::bitset<8>(bits[n]);
    std::cout << std::endl;
    
}

int main(int argc, char** argv)
{
    cuda_normals_2::Options opt(argc, argv);

    // Get the model
    ModelPtr model = ModelFactory::readModel(opt.inputFile());

    // Get the points
    PointBufferPtr pbuffer = model->m_pointCloud;
    size_t num_points = model->m_pointCloud->numPoints();
    size_t size =  3 * num_points;

    floatArr points = pbuffer->getPointArray();
    
    float* points_raw = &points[0];
    /*
    // Create the normal array
    floatArr normals(new float[size]);

    float* normals_raw = (float*) malloc(sizeof(float) * 3 * num_points);

    // Initialize the normals
    initNormals2(normals_raw, num_points);

    // Set the normal array
    pbuffer->setNormalArray(normals, num_points);
    */
    unsigned long long int* mortonCodes = (unsigned long long int*)
                    malloc(sizeof(unsigned long long int) * num_points);

    // Get the morton codes of the 3D points
    morton_codes_host(mortonCodes, points_raw, num_points);

    // Create an array which stores the id of each point
    int* point_IDs = (int*) malloc(sizeof(float) * num_points);

    for(int i = 0; i < num_points; i++)
    {
        point_IDs[i] = i;
    }
    // Sorting key-value pairs. point_IDs holds the sorted indices
    radix_sort(mortonCodes, point_IDs, num_points);

    // Save the new model as test.ply
    ModelFactory::saveModel(model, "test.ply");

    // free(normals_raw);
    free(mortonCodes);
    free(point_IDs);


    return 0;
}
