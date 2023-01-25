// #include <iostream>
// #include <fstream>
// #include <string>
// #include <cmath>
// #include <vector>

// std::vector<float> sphere_point_cloud(size_t num_points) {
//     std::vector<float> point_cloud;

//     // Generate points on a sphere
//     for (size_t i = 0; i < num_points; i++) {
//         float theta = 2 * M_PI * (i / static_cast<float>(num_points));
//         float phi = M_PI * (i / static_cast<float>(num_points));
//         float x = std::cos(theta) * std::sin(phi);
//         float y = std::sin(theta) * std::sin(phi);
//         float z = std::cos(phi);
//         point_cloud.emplace_back(x);
//         point_cloud.emplace_back(y);
//         point_cloud.emplace_back(z);
//     }

//     return point_cloud;
// }

// int main() {
//     size_t num_points = 1000000;
//     std::vector<float> pc = sphere_point_cloud(num_points);

//     // Open a file in write mode
//     std::ofstream out_file("sphere.pts");

//     // Check if the file was successfully opened
//     if (out_file.is_open()) {

//         for(int i = 0; i < num_points; i++)
//         {
//             out_file << pc[3 * i + 0] << " ";
//             out_file << pc[3 * i + 1] << " ";
//             out_file << pc[3 * i + 2] << " "; 
//             out_file << "0" << std::endl;
//         }

//         out_file.close();
//     } else {
//         std::cout << "Error opening file" << std::endl;
//     }

//     return 0;
// }




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

    // Create the normal array
    float* normals = (float*) malloc(sizeof(float) * num_queries * 3);

    int mode = 0;
    // #########################################################################################
    if(mode == 0)
    {
        tree.knn_normals(
            queries, 
            num_queries, 
            K,
            normals,
            num_queries
        );
    }

    //##########################################################################################
    if(mode == 1)
    {
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

        // Calculate the normals
        tree.calculate_normals(normals, num_queries,
                    queries, num_queries, K,
                    n_neighbors_out, indices_out);

        std::cout << n_neighbors_out[666] << std::endl;

    }

    // ########################################################################################
    // Set the normals in the Model
    floatArr new_normals = floatArr(&normals[0]);

    pbuffer->setNormalArray(new_normals, num_points);


    // for(int i = 0; i < 50; i++)
    // {
    //     std::cout << new_normals[i] << std::endl;
    // }
    std::cout << new_normals[0] << std::endl;
    std::cout << new_normals[555] << std::endl;
    std::cout << new_normals[100000] << std::endl;
    
    // Save the new model as test.ply
    ModelFactory::saveModel(model, "test.ply");

    return 0;
}
