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
    


    std::string kernel_file = "query_knn_kernels.cu";
    std::string kernel_name = "query_knn_kernel";

    std::string kernel_path = "../src/tools/lvr2_cuda_normals2/src/query_knn_kernels.cu";
    std::string cu_src = read_kernel(kernel_path);
    
    // size_t kernel_size = kernel_str.size();
    const char* kernel = cu_src.c_str();

    // std::cout << "Cuda Kernel: " << std::endl;
    // std::cout << kernel << std::endl;

    // // Create test points
    // num_points = 6;
    // float* test_points = (float*) malloc(sizeof(float) * num_points * 3);

    // test_points[0] = 0.0f;
    // test_points[1] = 1.0f;
    // test_points[2] = 2.0f;

    // test_points[3] = 100.0f;
    // test_points[4] = 99.0f;
    // test_points[5] = 101.0f;

    // test_points[6] = 25.0f;
    // test_points[7] = 26.0f;
    // test_points[8] = 27.0f;

    // test_points[9] = 0.5f;
    // test_points[10] = 1.5f;
    // test_points[11] = 2.25f;

    // test_points[12] = 111.0f;
    // test_points[13] = 101.0f;
    // test_points[14] = 100.0f;

    // test_points[15] = 40.0f;
    // test_points[16] = 45.5f;
    // test_points[17] = 35.25f;

    // // Create test queries
    
    // size_t num_queries = 3;

    // float* queries = (float*) malloc(sizeof(float) * 3 * 3);
    // queries[0] = 1.0f;
    // queries[1] = 2.0f;
    // queries[2] = 3.0f;

    // queries[3] = 101.0f;
    // queries[4] = 100.0f;
    // queries[5] = 102.0f;

    // queries[6] = 26.0f;
    // queries[7] = 27.0f;
    // queries[8] = 28.0f;

    size_t num_queries = num_points;

    float* queries = points_raw;

    // float* queries = (float*) malloc(sizeof(float) * 3 * num_queries);
    // queries[0] = 0.0f;
    // queries[1] = 0.0f;
    // queries[2] = 0.0f;

    // queries[3] = 50.0f;
    // queries[4] = 50.0f;
    // queries[5] = 50.0f;

    // queries[6] = 20.0f;
    // queries[7] = 20.0f;
    // queries[8] = 20.0f;

    // queries[9] = 150.0f;
    // queries[10] = 150.0f;
    // queries[11] = 150.0f;
    

    float* args = (float*) malloc(sizeof(float));
    args[0] = 1.0f;

    // the normals will be stored here
    float* normals = (float*) malloc(sizeof(float) * num_queries * 3);

    build_lbvh(points_raw, num_points, queries, num_queries, args, kernel, kernel_name.c_str(), normals);

    floatArr new_normals = floatArr(&normals[0]);

    pbuffer->setNormalArray(new_normals, num_points);

    std::cout << new_normals[0] << std::endl;

    // int point_idx = 5875;

    // std::cout << "x: " << normals[3 * point_idx + 0] << std::endl;
    // std::cout << "y: " << normals[3 * point_idx + 1] << std::endl;
    // std::cout << "z: " << normals[3 * point_idx + 2] << std::endl;

    // Save the new model as test.ply
    ModelFactory::saveModel(model, "test.ply");

    // free(normals_raw);
    // free(mortonCodes);
    // free(point_IDs);

    return 0;
}
