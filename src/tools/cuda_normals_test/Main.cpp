#include <iostream>

#include "lvr2/types/PointBuffer.hpp"

#include "lvr2/reconstruction/cuda/LBVHIndex.hpp"

#include "lvr2/util/Synthetic.hpp"

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
        5000000, 
        // 10000000, 
        // 20000000, 
        // 30000000
    };

    int k_s[] = {
        10, 
        25, 
        50, 
        100, 
        200
    };

    // Worked with n = 1000000, k = 200

    // Generates 36.000.000 points
    lvr2::PointBufferPtr pbuffer;
    pbuffer = lvr2::synthetic::genSpherePoints(6000,6000);

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
        tree.build(pts, n);

        for(int k : k_s)
        {
            std::cout << "Testing with k = " << k << std::endl;
            tree.knn_normals(
                pts,
                n,
                k,
                normals,
                n
            );
        }
    }
}