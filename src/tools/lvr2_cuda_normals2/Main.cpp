#include <boost/filesystem.hpp>
#include <iostream>
#include <bitset>
#include <vector>
#include <fstream>
#include <string>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "Options.hpp"

#include "LBVHIndex.cuh"

using namespace lvr2;

void floatToBinary(float f)
{
    char* bits = reinterpret_cast<char*>(&f);
    for(int n = sizeof(f) - 1; n >= 0; n--)
        std::cout << std::bitset<8>(bits[n]);
    std::cout << std::endl;
    
}

// std::string read_kernel(std::string file_path)
// {
//     std::ifstream in(file_path);
//     std::string contents((std::istreambuf_iterator<char>(in)),
//         std::istreambuf_iterator<char>());

//     return contents;
// }

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
    
    // Create the LBVH
    int leaf_size = 1;
    bool sort_queries = true;
    bool compact = true;

    int K = 50;

    lbvh::LBVHIndex tree(leaf_size, sort_queries, compact);

    tree.build(points_raw, num_points);


    // // Get the Query Kernel
    // std::string kernel_file = "query_knn_kernels.cu";
    // std::string kernel_name = "query_knn_kernel";

    // std::string kernel_path = "../src/tools/lvr2_cuda_normals2/src/query_knn_kernels.cu";
    // std::string cu_src = read_kernel(kernel_path);
    
    // const char* kernel = cu_src.c_str();

    // Get the queries
    size_t num_queries = num_points;

    float* queries = points_raw;

    // Create the return arrays
    unsigned int* n_neighbors_out;
    unsigned int* indices_out;
    float* distances_out;

    // Malloc the output arrays here
    n_neighbors_out = (unsigned int*) malloc(sizeof(unsigned int) * num_queries);
    indices_out = (unsigned int*) malloc(sizeof(unsigned int) * num_queries * K);
    distances_out = (float*) malloc(sizeof(float) * num_queries * K);

    // Process the queries 
    tree.kSearch(queries, num_queries,
                K,
                n_neighbors_out, indices_out, distances_out);


    // Create the normal array
    float* normals = (float*) malloc(sizeof(float) * num_queries * 3);

    // Calculate the normals
    tree.calculate_normals(normals, num_queries,
                queries, num_queries, K,
                n_neighbors_out, indices_out);

    // Set the normals in the Model
    floatArr new_normals = floatArr(&normals[0]);

    pbuffer->setNormalArray(new_normals, num_points);

    std::cout << new_normals[0] << std::endl;

    // Save the new model as test.ply
    ModelFactory::saveModel(model, "test.ply");

    return 0;
}
