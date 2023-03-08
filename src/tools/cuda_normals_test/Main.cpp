#include <iostream>
#include <chrono>
#include <iostream>
#include <fstream>

#include "lvr2/types/PointBuffer.hpp"

#include "lvr2/reconstruction/cuda/LBVHIndex.hpp"

#include "lvr2/util/Synthetic.hpp"

#include "lvr2/reconstruction/SearchTreeLBVH.hpp"

int main() 
{
    lvr2::lbvh::LBVHIndex tree = lvr2::lbvh::LBVHIndex(1, true, true);

    size_t n_s[] = {
        // 100, 
        // 1000, 
        // 10000, 
        // 100000,
        // 500000, 
        // 1000000, 
        // 5000000, 
        // 10000000, 
        // 20000000,
        // 25000000 
        // 30000000
        // 30000000
        35000000
        // 40000000
    };

    int k_s[] = {
        10, 
        // 25, 
        // 50, 
        // 100, 
        // 200,
        // 500,
        // 1000,
        // 2000,
        // 5000,
        // 10000
    };

    const char *path = "../src/tools/cuda_normals_test";
    std::ofstream myfile(path);
    myfile.open("runtime_test.txt");

    // Worked with n = 5000000, k = 200
    // Worked with n = 10000000, k = 100
    // Worked with n = 20000000, k = 25
    // Worked with n= 30000000, k = 10
    // Failed for n = 10000000 K = 10

    // Generates 36.000.000 points
    lvr2::PointBufferPtr pbuffer;
    pbuffer = lvr2::synthetic::genSpherePoints(8000,8000);

    size_t num_points = pbuffer->numPoints();

    lvr2::floatArr points = pbuffer->getPointArray();
    float* points_raw = &points[0];

    for(size_t n : n_s)
    {
        // Get n points
        float* pts = (float*) malloc(sizeof(float) * n * 3);
        
        float* normals = (float*) malloc(sizeof(float) * n * 3);

        for(size_t i = 0; i < n; i++)
        {
            pts[3 * i + 0] = points_raw[3 * i + 0];
            pts[3 * i + 1] = points_raw[3 * i + 1];
            pts[3 * i + 2] = points_raw[3 * i + 2];

        }
        std::cout << "Building LBVH with n = " << n << std::endl;
        myfile << "Building LBVH with n = " << n << std::endl;
        std::chrono::steady_clock::time_point begin_build = std::chrono::steady_clock::now();
        tree.build(pts, n);
        
        std::chrono::steady_clock::time_point end_build = std::chrono::steady_clock::now();

        std::cout << "Time Building Tree: " << std::chrono::duration_cast<std::chrono::milliseconds> (end_build - begin_build).count() << "[ms]" << std::endl;
        myfile << "Time Building Tree: " << std::chrono::duration_cast<std::chrono::milliseconds> (end_build - begin_build).count() << "[ms]" << std::endl;

        int mode = 0;
        std::cout << "MODE " << mode << std::endl;
        myfile << "MODE " << mode << std::endl;

        for(int k : k_s)
        {   
            std::cout << "Testing with k = " << k << std::endl;
            myfile << "Testing with k = " << k << std::endl;

            std::chrono::steady_clock::time_point begin_knn = std::chrono::steady_clock::now();
            
            if(mode == 0)
            {
                
                tree.knn_normals(
                    // pts,
                    // n,
                    k,
                    normals,
                    n
                );

            }

            if(mode == 1)
            {

                // Create the return arrays
                unsigned int* n_neighbors_out;
                unsigned int* indices_out;
                float* distances_out;

                // Malloc the output arrays here
                n_neighbors_out = (unsigned int*) malloc(sizeof(unsigned int) * n);
                indices_out = (unsigned int*) malloc(sizeof(unsigned int) * n * k);
                distances_out = (float*) malloc(sizeof(float) * n * k);

                // Process the queries 
                tree.kSearch(
                    pts, 
                    n,
                    k,
                    n_neighbors_out, 
                    indices_out, 
                    distances_out);
            
               
                // Calculate the normals
                tree.calculate_normals(
                    normals, 
                    n,
                    pts, 
                    n,
                    k,
                    n_neighbors_out, 
                    indices_out);
            }
        
            std::chrono::steady_clock::time_point end_knn = std::chrono::steady_clock::now();
            std::cout << "Time Calculating Normals: " << std::chrono::duration_cast<std::chrono::milliseconds> (end_knn - begin_knn).count() << "[ms]" << std::endl;
            myfile << "Time Calculating Normals: " << std::chrono::duration_cast<std::chrono::milliseconds> (end_knn - begin_knn).count() << "[ms]" << std::endl;
            std::cout << std::endl;
            myfile << std::endl;
        }
        myfile << std::endl;
    }

    myfile.close();
    return 0;
}