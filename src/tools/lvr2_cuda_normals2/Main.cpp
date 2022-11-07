#include <boost/filesystem.hpp>
#include <iostream>
#include <bitset>
#include <vector>
#include <fstream>
#include <string>

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

std::string read_kernel(std::string file_path)
{
    std::ifstream in(file_path);
    std::string contents((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    return contents;
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

    // unsigned long long int* mortonCodes = (unsigned long long int*)
    //                 malloc(sizeof(unsigned long long int) * num_points);

    // // Get the morton codes of the 3D points
    // morton_codes_host(mortonCodes, points_raw, num_points);

    // // Create an array which stores the id of each point
    // int* point_IDs = (int*) malloc(sizeof(float) * num_points);

    // for(int i = 0; i < num_points; i++)
    // {
    //     point_IDs[i] = i;
    // }
    // // Sorting key-value pairs. point_IDs holds the sorted indices
    // radix_sort(mortonCodes, point_IDs, num_points);
    
    // Read in query_knn_kernels.cu
    
    std::string kernel_str;

    kernel_str = read_kernel("../src/tools/lvr2_cuda_normals2/src/query_knn_kernels.cu");
    
    // size_t kernel_size = kernel_str.size();
    const char* kernel = kernel_str.c_str();

    std::cout << "Cuda Kernel: " << std::endl;
    std::cout << kernel << std::endl;


    // Create test queries
    
    size_t num_queries = 1;

    float* queries = (float*) malloc(sizeof(float) * 3);
    queries[0] = 1.5f;
    queries[1] = 1.5f;
    queries[2] = 1.5f;

    float* args = (float*) malloc(sizeof(float));
    args[0] = 1.0f;

    build_lbvh(points_raw, num_points, queries, num_queries, args, kernel);

    // Save the new model as test.ply
    ModelFactory::saveModel(model, "test.ply");

    // free(normals_raw);
    // free(mortonCodes);
    // free(point_IDs);

    return 0;
}
