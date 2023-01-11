#include "lvr2/reconstruction/SearchTreeLBVH.hpp"

namespace lvr2
{

template<typename BaseVecT>
SearchTreeLBVH<BaseVecT>::SearchTreeLBVH(PointBufferPtr pbuffer)
{
    m_tree = lbvh::LBVHIndex(1, true, true);
    
    float* points = &pbuffer->getPointArray()[0];
    size_t num_points = pbuffer->numPoints();

    m_tree.build(points, num_points);

}

template<typename BaseVecT>
int SearchTreeLBVH<BaseVecT>::kSearch(
    const BaseVecT& qp,
    int K,
    vector<size_t>& indices,
    vector<CoordT>& distances
) const
{
    float query_point[] = {qp.x, qp.y, qp.z};
    size_t num_queries = 1;

    indices.resize(K);
    distances.resize(K);

    // Create the return arrays
    unsigned int* n_neighbors_out;
    unsigned int* indices_out;
    float* distances_out;

    n_neighbors_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries);
    indices_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries * K);
    distances_out = (float*) 
        malloc(sizeof(float) * num_queries * K);

    // Perform the knn search
    m_tree.kSearch(
        query_point, num_queries,
        K, 
        n_neighbors_out, 
        indices_out, 
        distances_out
    );

    for(int i = 0; i < K; i++)
    {
        indices.push_back(indices_out[i]);
        distances.push_back(distances_out[i]);
    }

    return n_neighbors_out[0];
}

template<typename BaseVecT>
int SearchTreeLBVH<BaseVecT>::radiusSearch(
    const BaseVecT& qp,
    int K,
    float r,
    vector<size_t>& indices,
    vector<CoordT>& distances
) const 
{
    float query_point[] = {qp.x, qp.y, qp.z};
    size_t num_queries = 1;

    // Create the return arrays
    unsigned int* n_neighbors_out;
    unsigned int* indices_out;
    float* distances_out;

    n_neighbors_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries);
    indices_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries * K);
    distances_out = (float*) 
        malloc(sizeof(float) * num_queries * K);

    // Perform the knn search
    m_tree.radiusSearch(
        query_point, num_queries,
        K, r,
        n_neighbors_out, 
        indices_out, 
        distances_out
    );

    size_t n = n_neighbors_out[0];

    indices.resize(n);
    distances.resize(n);

    for(int i = 0; i < n; i++)
    {
        indices.push_back(indices_out[i]);
        distances.push_back(distances_out[i]);
    }

    return n_neighbors_out[0];
}

template<typename BaseVecT>
void SearchTreeLBVH<BaseVecT>::kSearchParallel(
    const BaseVecT* queries,
    int num_queries,
    int K,
    vector<size_t>& indices,
    vector<CoordT>& distances
) const
{
    // float* query_points = &queries[0];

    // TODO Do this in a different way
    float* query_points = 
        (float*) malloc(sizeof(float) * 3 * num_queries);

    for(int i = 0; i < num_queries; i++)
    {
        query_points[3 * i + 0] = queries[i].x;
        query_points[3 * i + 1] = queries[i].y;
        query_points[3 * i + 2] = queries[i].z;
    }

    // Create the return arrays
    unsigned int* n_neighbors_out;
    unsigned int* indices_out;
    float* distances_out;

    n_neighbors_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries);
    indices_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries * K);
    distances_out = (float*) 
        malloc(sizeof(float) * num_queries * K);

    // Perform the knn search
    m_tree.kSearch(
        query_points, 
        num_queries,
        K,
        n_neighbors_out, 
        indices_out, 
        distances_out
    );

    indices.resize(num_queries * K);
    distances.resize(num_queries * K);

    for(int i = 0; i < num_queries; i++)
    {
        for(int j = 0; j < K; j++)
        {
            indices.push_back(indices_out[i * K + j]);
            distances.push_back(distances_out[i * K + j]);
        }
    }

}

template<typename BaseVecT>
void SearchTreeLBVH<BaseVecT>::radiusSearchParallel(
    const BaseVecT* query,
    int num_queries,
    int K,
    float r,
    vector<size_t>& indices,
    vector<CoordT>& distances,
    vector<unsigned int>& neighbors
) const
{
    float* query_points = &query[0];

    // Create the return arrays
    unsigned int* n_neighbors_out;
    unsigned int* indices_out;
    float* distances_out;

    n_neighbors_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries);
    indices_out = (unsigned int*) 
        malloc(sizeof(unsigned int) * num_queries * K);
    distances_out = (float*) 
        malloc(sizeof(float) * num_queries * K);

    // Perform the knn search
    m_tree.radiusSearch(
        query_points, num_queries,
        K, r,
        n_neighbors_out, 
        indices_out, 
        distances_out
    );

    for(int i = 0; i < num_queries; i++)
    {
        for(int j = 0; j < n_neighbors_out[i]; j++)
        {
            indices.push_back(indices_out[i * K + j]);
            distances.push_back(distances_out[i * K + j]);
        }
        neighbors.push_back(n_neighbors_out[i]);
    }
}



} // namespace lvr2