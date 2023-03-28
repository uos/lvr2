/**
 * Copyright (c) 2023, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LVR2_LBVHINDEX_HPP
#define LVR2_LBVHINDEX_HPP

#include <string>

namespace lvr2
{

namespace lbvh
{

struct AABB;        // Forward Declaration
struct BVHNode;     // Forward Declaration

/**
 * @brief   A class that uses a GPU based approach to find nearest neighbors in a data set
 *          and calculate surface normals on the neighborhoods by using a LBVH as acceleration
 *          structure for the kNN search
 */
class LBVHIndex
{
public:
    // CPU
    unsigned int m_num_objects; // Number of points in the dataset (= n)
    unsigned int m_num_nodes;   // Number of nodes in the LBVH (2n-1)
    unsigned int m_leaf_size;   // Max number of points covered by a leaf node
    bool m_sort_queries;        // True, if queries should be sorted by their morton codes
    bool m_compact;             // True, if the tree should be compacted after the optimization step

    unsigned int* m_root_node;  // The root node of the LBVH 

    float m_flip_x;             // x coordinate that the normals will be flipped to
    float m_flip_y;             // y coordinate that the normals will be flipped to
    float m_flip_z;             // z coordinate that the normals will be flipped to

    // GPU
    float* m_d_points;                  // Points in the dataset, stored as device pointer
    unsigned int* m_d_sorted_indices;   // Sorted indices of the points, stored as device pointer 
    BVHNode* m_d_nodes;                 // Nodes of the LBVH, stored as device pointer
    AABB* m_d_extent;                   // Extent of the dataset (min and max value in each dimension)

    /**
     * @brief Default constructor
     *      m_num_objects = 0;
     *      m_num_nodes = 0;
     *      m_leaf_size = 1;
     *      m_sort_queries = true;
     *      m_compact = true;
     *      m_flip_x = 1000000.0;
     *      m_flip_y = 1000000.0;
     *      m_flip_z = 1000000.0;
     */
    LBVHIndex();

    /**
     * @brief Constructor of the LBVH
     * 
     * @param leaf_size     Max number of points covered by a leaf node
     * @param sort_queries  True, if queries should be sorted by their
     *                      morton codes
     * @param compact       True, if the tree should be compacted after
     *                      the optimization step
     * @param flip_x        x coordinate that the normals will be flipped to
     * @param flip_y        y coordinate that the normals will be flipped to
     * @param flip_z        z coordinate that the normals will be flipped to
     */
    LBVHIndex(
        int leaf_size, 
        bool sort_queries, 
        bool compact,
        float flip_x=1000000.0f, 
        float flip_y=1000000.0f, 
        float flip_z=1000000.0f
    );

    /**
     * @brief Destructor of the LBVH
     */
    ~LBVHIndex();

    /**
     * @brief   This function builds the LBVH on the given points
     * 
     * @param points        The points of the dataset
     * @param num_points    The number of points in the dataset
     */
    void build(
        float* points, size_t num_points
    );
    
    /**
     * @brief   This function performs a kNN search on the LBVH
     *          with the given queries
     * 
     * @param query_points      The query points.
     * @param num_queries       Number of query points
     * @param K                 The number of neighbours that should be searched.
     * @param n_neighbors_out   Stores the number of found neighbors
     *                          for each query
     * @param indices_out       Stores the indices of all found neighbors
     * @param distances_out     Stores the distances of all found neighbors
     */
    void kSearch(
        float* query_points, 
        size_t num_queries,
        int K, 
        unsigned int* n_neighbors_out, 
        unsigned int* indices_out, 
        float* distances_out
    ) const;
    
    /**
     * @brief   This function performs a kNN search on the LBVH
     *          with the given queries, but expects device pointers
     * 
     * @param d_query_points    The query points as device pointer
     * @param num_queries       Number of query points
     * @param K                 The number of neighbours that should be searched.
     * @param d_n_neighbors_out Stores the number of found neighbors
     *                          for each query as device pointer
     * @param d_indices_out     Stores the indices of all found neighbors
     *                          as device pointer
     * @param d_distances_out   Stores the distances of all found neighbors
     *                          as device pointer
     */
    void kSearch_dev_ptr(
        float* d_query_points, 
        size_t num_queries,
        int K, 
        unsigned int* d_n_neighbors_out, 
        unsigned int* d_indices_out, 
        float* d_distances_out
    ) const;

    /**
     * @brief   This function performs a radius search on the LBVH
     *          with the given queries
     * 
     * @param query_points      The query points.
     * @param num_queries       Number of query points
     * @param K                 The number of neighbours that should be searched.
     * @param r                 The radius
     * @param n_neighbors_out   Stores the number of found neighbors
     *                          for each query
     * @param indices_out       Stores the indices of all found neighbors
     * @param distances_out     Stores the distances of all found neighbors
     */
    void radiusSearch(
        float* query_points, 
        size_t num_queries,
        int K, 
        float r,
        unsigned int* n_neighbors_out, 
        unsigned int* indices_out, 
        float* distances_out
    ) const;
    
    /**
     * @brief   This function performs a radius search on the LBVH
     *          with the given queries, but expects device pointers
     * 
     * @param d_query_points    The query points.
     * @param num_queries       Number of query points
     * @param K                 The number of neighbours that should be searched.
     * @param r                 The radius
     * @param d_n_neighbors_out Stores the number of found neighbors
     *                          for each query as device pointer
     * @param d_indices_out     Stores the indices of all found neighbors
     *                          as device pointers
     * @param d_distances_out   Stores the distances of all found neighbors
     *                          as device pointers
     */
    void radiusSearch_dev_ptr(
        float* d_query_points, 
        size_t num_queries,
        int K, 
        float r,
        unsigned int* d_n_neighbors_out, 
        unsigned int* d_indices_out, 
        float* d_distances_out
    ) const;

    /**
     * @brief This function processes the queries by calling the query_knn_kernel
     * 
     * @param queries_raw       The query points.
     * @param num_queries       Number of query points
     * @param K                 The number of neighbours that should be searched.
     * @param r                 The radius
     * @param n_neighbors_out   Stores the number of found neighbors
     *                          for each query
     * @param indices_out       Stores the indices of all found neighbors
     * @param distances_out     Stores the distances of all found neighbors   
     */
    void process_queries(
        float* queries_raw, 
        size_t num_queries, 
        int K, 
        float r,
        unsigned int* n_neighbors_out, 
        unsigned int* indices_out, 
        float* distances_out
    ) const;

    /**
     * @brief   This function processes the queries by calling the query_knn_kernel,
     *          but expects device pointer
     * 
     * @param d_query_points    The query points.
     * @param num_queries       Number of query points
     * @param K                 The number of neighbours that should be searched.
     * @param r                 The radius
     * @param d_n_neighbors_out Stores the number of found neighbors
     *                          for each query as device pointer
     * @param d_indices_out     Stores the indices of all found neighbors 
     *                          as device pointers
     * @param d_distances_out   Stores the distances of all found neighbors   
     *                          as device pointers
     */
    void process_queries_dev_ptr(
        float* d_query_points, 
        size_t num_queries,
        int K, 
        float r,
        unsigned int* d_n_neighbors_out, 
        unsigned int* d_indices_out, 
        float* d_distances_out
    ) const;

    /**
     * @brief   This function calculates the normals of each query
     *          with the given nearest neighborhoods by calling the 
     *          calculate_normals_kernel
     * 
     * @param normals           Stores the calculated normals
     * @param num_normals       Number of normals that are calculated
     * @param queries           The queries for which the normal is calculated
     * @param num_queries       Number of queries
     * @param K                 The number of neighbours that should be searched.
     * @param n_neighbors_in    Number of found neighbors for each query
     * @param indices_in        Indices of the found neighbors
     */
    void calculate_normals(
        float* normals, 
        size_t num_normals,
        float* queries,    
        size_t num_queries,
        int K,
        const unsigned int* n_neighbors_in, 
        const unsigned int* indices_in
    ) const;

    /**
     * @brief   This function performs a kNN search on each point in the dataset
     *          and calculates the normals on the neighborhoods in a single
     *          kernel launch (knn_normals_kernel)
     * 
     * @param K             The number of neighbours that should be searched.
     * @param normals       Stores the calculated normals
     * @param num_normals   Number of normals to be calculated
     */
    void knn_normals(
        int K,
        float* normals, 
        size_t num_normals
    );

    /**
     * @brief   This function gets the extent of the dataset
     *          (max and min values of each dimension)
     *  
     * @param extent        Stores the extent of the dataset
     * @param points        The points of the dataset
     * @param num_points    Number of points in the dataset
     */
    void getExtent(
        AABB* extent, 
        float* points, 
        size_t num_points
    ) const;

    /**
     * @brief   This function creates a PTX from a given CUDA file
     * 
     * @param ptx           String that stores the PTX
     * @param sample_name   Name of the CUDA kernel 
     * @param cu_source     The CUDA file as char*
     * @param K             The number of neighbours that should be searched.
     */
    void getPtxFromCuString( 
        std::string& ptx, 
        const char* sample_name, 
        const char* cu_source, 
        int K
    ) const;

};

} // namespace lbvh

} // namespace lvr2

#endif // LVR2_LBVHINDEX_HPP