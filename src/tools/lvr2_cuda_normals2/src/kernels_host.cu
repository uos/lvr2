#include "kernels_host.h"

#include "LBVHIndex.cuh"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <vector>

#include <opencv2/core.hpp>

using namespace lbvh;
using namespace cv;

void build_lbvh(float* points, size_t num_points,
                float* queries, size_t num_queries,
                float* args,
                const char* kernel, const char* kernel_name,
                float* normals)
{
    int size_points = num_points * 3 * sizeof(float);

    int leaf_size = 1;
    bool sort_queries = true;
    bool compact = true;
    bool shrink_to_fit = true;

    int K = 50;

    lbvh::LBVHIndex tree(leaf_size, sort_queries, compact, shrink_to_fit);

    tree.build(points, num_points);
   
    unsigned int* n_neighbors_out = (unsigned int*)
                malloc(sizeof(unsigned int) * num_queries);

    unsigned int* indices_out = (unsigned int*) 
                malloc(sizeof(unsigned int) * num_queries * K);

    float* distances_out = (float*)
                malloc(sizeof(float) * num_queries * K);

    // TODO: Don't process the queries here
    tree.process_queries(queries, num_queries, args, points, num_points, 
                        kernel, kernel_name, K,
                        n_neighbors_out, indices_out, distances_out);

    tree.calculate_normals(normals, num_queries,
                queries, num_queries, K, points, num_points,
                n_neighbors_out, indices_out);


    // CPU NORMALS ###############################################################################################################

    // // Perform PCA to get the normal for each point
    // int idx_cnt = 0;      // Count the number of neighbors iterated through. Important for indexing when using radius search


    // // bool start_no_neigh = false;    // Debugging TODO: Delete
    // // Iterate through each point
    // for(int i = 0; i < num_queries; i++)
    // {
    //     // Create a matrix with all neighbor points of point i
    //     Mat m = Mat::zeros(n_neighbors_out[i], 3, CV_32F);

    //     // iterate through each neighbor of point i
    //     for(int k = 0; k < n_neighbors_out[i]; k++)
    //     {
    //         unsigned int neigh_idx = k + idx_cnt;

    //         // Iterate through each dimension
    //         for(int x = 0; x < 3; x++)
    //         {
    //             m.at<float>(k, 0) = points[ 3 * indices_out[neigh_idx] + 0];
    //             m.at<float>(k, 1) = points[ 3 * indices_out[neigh_idx] + 1];
    //             m.at<float>(k, 2) = points[ 3 * indices_out[neigh_idx] + 2];
    //         }
            
    //     }

    //     // TODO: Do the PCA
    //     PCA pca_analysis(m, Mat(), PCA::DATA_AS_ROW);

       
    //     std::vector<float> eigen_vals = pca_analysis.eigenvalues;

    //     unsigned int min_val_idx = 0;
            
    //     for(int e = 1; e < eigen_vals.size(); e++)
    //     {
    //         if(eigen_vals[e] < eigen_vals[min_val_idx])
    //         {
    //             min_val_idx = e;
    //         }
    //     }

    //     normals[3 * i + 0] = pca_analysis.eigenvectors.at<float>(min_val_idx, 0);
    //     normals[3 * i + 1] = pca_analysis.eigenvectors.at<float>(min_val_idx, 1);
    //     normals[3 * i + 2] = pca_analysis.eigenvectors.at<float>(min_val_idx, 2);


    //     idx_cnt += n_neighbors_out[i];

    // }
    // #################################################################################################


    // int idx = num_queries - 1;

    // std::cout << "Neighbors: " << std::endl;
    // std::cout << points[ 3 * indices_out[ 3 * idx + 0] + 0] << ", "
    //           << points[ 3 * indices_out[ 3 * idx + 0] + 1] << ", "
    //           << points[ 3 * indices_out[ 3 * idx + 0] + 2] << "; "
    //           << std::endl;
    
    // std::cout << points[ 3 * indices_out[ 3 * idx + 1] + 0] << ", "
    //           << points[ 3 * indices_out[ 3 * idx + 1] + 1] << ", "
    //           << points[ 3 * indices_out[ 3 * idx + 1] + 2] << "; "
    //           << std::endl;

    // std::cout << points[ 3 * indices_out[ 3 * idx + 2] + 0] << ", "
    //           << points[ 3 * indices_out[ 3 * idx + 2] + 1] << ", "
    //           << points[ 3 * indices_out[ 3 * idx + 2] + 2] << "; "
    //           << std::endl;



    // Mat m = Mat::zeros(3, 5, CV_32F);

    // m.at<float>(0,0) = 1.0f;

    // std::cout << "m = \n" << m.row(0) << std::endl;

    

    // std::cout << "Number of neighbors: " << std::endl;
    // for(int i = 0; i < num_queries; i++)
    // {
    //     std::cout << n_neighbors_out[i] << std::endl;
    // }

    // std::cout << "Neighbor Index: " << std::endl;

    // for(int i = 0; i < num_queries * K; i++)
    // {
    //     std::cout << indices_out[i] << std::endl;
    // }

    // std::cout << "Distances Out: " << std::endl;

    // for(int i = 0; i < num_queries * K; i++)
    // {
    //     std::cout << distances_out[i] << std::endl;
    // }
    

    return;
}