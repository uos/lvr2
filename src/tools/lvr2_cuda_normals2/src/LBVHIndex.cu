#include "LBVHIndex.cuh"

using namespace lbvh;

LBVHIndex::LBVHIndex(int leaf_size, bool sort_queries, bool compact, bool shrink_to_fit)
{
    m_num_objects = -1;
    m_num_nodes = -1;
    m_leaf_size = leaf_size;
    m_sort_queries = sort_queries;
    m_compact = compact;
    m_shrink_to_fit = shrink_to_fit;

}


void LBVHIndex::build(float* points, size_t num_points)
{
    m_points = points;

    m_num_objects = num_points;
    m_num_nodes = 2 * m_num_objects - 1;

    // initialize AABBs
    AABB* aabbs[num_points];

    // Initial bounding boxes are the points
    for(int i = 0; i < 3 * m_num_objects; i += 3)
    {
        aabbs[i]->min.x = points[i + 0];
        aabbs[i]->max.x = points[i + 0];
        aabbs[i]->min.y = points[i + 1];
        aabbs[i]->max.y = points[i + 1];
        aabbs[i]->min.z = points[i + 2];
        aabbs[i]->max.z = points[i + 2];
    }
    // Get the extent
    AABB extent = getExtent(m_points, m_num_objects);

    // Get the morton codes of the points
    unsigned long long int* mortonCodes = (unsigned long long int*)
                    malloc(sizeof(unsigned long long int) * m_num_objects);

    
}

// Get the extent of the points 
// (minimum and maximum values in each dimension)
__device__ __host__ AABB LBVHIndex::getExtent(float* points, size_t num_points)
{
    float min_x = INT_MAX;
    float min_y = INT_MAX;
    float min_z = INT_MAX;

    float max_x = INT_MIN; 
    float max_y = INT_MIN; 
    float max_z = INT_MIN;

    for(int i = 0; i < 3 * num_points; i += 3)
    {
        if(points[i + 0] < min_x)
        {
            min_x = points[i + 0];
        }

        if(points[i + 1] < min_y)
        {
            min_y = points[i + 1];
        }

        if(points[i + 2] < min_z)
        {
            min_z = points[i + 2];
        }

        if(points[i + 0] > max_x)
        {
            max_x = points[i + 0];
        }

        if(points[i + 1] > max_y)
        {
            max_y = points[i + 1];
        }

        if(points[i + 2] > max_z)
        {
            max_z = points[i + 2];
        }
    }
    
    AABB extent;
    extent.min.x = min_x;
    extent.min.y = min_y;
    extent.min.z = min_z;
    
    extent.max.x = max_x;
    extent.max.y = max_y;
    extent.max.z = max_z;
    
    return extent;
}