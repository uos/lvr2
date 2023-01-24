#include "SearchTreeLBVH.hpp"

#include <iostream>

#define NUM 5000000

// TODO Only for testing
#include <cmath>
#include <vector>
#include <chrono>

std::vector<float> sphere_point_cloud(size_t num_points) {
    std::vector<float> point_cloud;

    // Generate points on a sphere
    for (size_t i = 0; i < num_points; i++) {
        float theta = 2 * M_PI * (i / static_cast<float>(num_points));
        float phi = M_PI * (i / static_cast<float>(num_points));
        float x = std::cos(theta) * std::sin(phi);
        float y = std::sin(theta) * std::sin(phi);
        float z = std::cos(phi);
        point_cloud.emplace_back(x);
        point_cloud.emplace_back(y);
        point_cloud.emplace_back(z);
    }

    return point_cloud;
}

namespace lvr2
{

template<typename BaseVecT>
CudaKSearchSurface<BaseVecT>::CudaKSearchSurface()
{
    this->setKn(50);

}

template<typename BaseVecT>
CudaKSearchSurface<BaseVecT>::CudaKSearchSurface(
    PointBufferPtr pbuffer,
    size_t k
) : PointsetSurface<BaseVecT>(pbuffer), m_tree(1, true, true)  //TODO leaf_size??
{
    this->setKn(k);

    floatArr points = this->m_pointBuffer->getPointArray();
    float* points_raw = &points[0];
    int num_points = this->m_pointBuffer->numPoints();
    // size_t num_points = NUM;
    // std::vector<float> pc = sphere_point_cloud(num_points);
    // float* points_raw = (float*) malloc(sizeof(float) * 3 * num_points);

    // for(int i = 0; i < num_points; i++)
    // {
    //     points_raw[3 * i + 0] = pc[3 * i + 0];
    //     points_raw[3 * i + 1] = pc[3 * i + 1];
    //     points_raw[3 * i + 2] = pc[3 * i + 2];
    // }
    std::cout << "Building Tree..." << std::endl;
    this->m_tree.build(points_raw, num_points);
    std::cout << "Tree Done!" << std::endl;

}

template<typename BaseVecT>
std::pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>
    CudaKSearchSurface<BaseVecT>::distance(BaseVecT p) const
{
    // Not implemented here
    throw std::runtime_error("Do not use this function!");
    return std::pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>(0.0, 0.0);
}

template<typename BaseVecT>
void CudaKSearchSurface<BaseVecT>::calculateSurfaceNormals()
{
    floatArr points = this->m_pointBuffer->getPointArray();
    
    float* points_raw = &points[0];
    int num_points = this->m_pointBuffer->numPoints();
    // int num_points = NUM;
    // float* points_raw = (float*) malloc(sizeof(float) * 3 * num_points);

    // for(int i = 0; i < num_points; i++)
    // {
    //     points_raw[i] = (float) i;
    // }
    // TODO Check if 3 floats long
    BaseVecT* query = reinterpret_cast<BaseVecT*>(points_raw);

    int K = this->m_kn;

    size_t size =  3 * num_points;

    // Get the queries
    size_t num_queries = num_points;

    float* queries = points_raw;

    // Create the normal array
    float* normals = (float*) malloc(sizeof(float) * num_queries * 3);

    int mode = 0;
    // #########################################################################################
    if(mode == 0)
    {

        std::cout << "KNN & Normals..." << std::endl;
        this->m_tree.knn_normals(
            queries, 
            num_queries, 
            K,
            normals,
            num_queries
        );
        std::cout << "Done!" << std::endl;
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

        std::cout << "KNN Search..." << std::endl;
        // Process the queries 
        this->m_tree.kSearch(queries, num_queries,
                    K,
                    n_neighbors_out, indices_out, distances_out);
       
        std::cout << "Normals..." << std::endl;
        // Calculate the normals
        this->m_tree.calculate_normals(normals, num_queries,
                    queries, num_queries, K,
                    n_neighbors_out, indices_out);

        std::cout << "Done!" << std::endl;

    }

    // ########################################################################################
    // Set the normals in the point buffer
    floatArr new_normals = floatArr(&normals[0]);

    this->m_pointBuffer->setNormalArray(new_normals, num_points);

}

} // namespace lvr2