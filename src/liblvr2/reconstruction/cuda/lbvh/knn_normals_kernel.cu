// #include "query_knn.cuh"

// using namespace lvr2;
// using namespace lbvh;

// namespace lvr2
// {

// extern "C" __global__ void knn_normals_kernel(
//     const BVHNode *nodes,
//     const float* __restrict__ points,         
//     const unsigned int* __restrict__ sorted_indices,
//     const unsigned int root_index,
//     const float max_radius,
//     const float* __restrict__ query_points,    
//     const unsigned int* __restrict__ sorted_queries,
//     const unsigned int num_queries, 
//     float* normals
// )
// {
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid >= num_queries)
//     {
//         return;
//     }

//     float flip_x=1000000.0; 
//     float flip_y=1000000.0; 
//     float flip_z=1000000.0;

//     StaticPriorityQueue<float, K> queue(max_radius);
//     unsigned int query_idx = sorted_queries[tid];

//     normals[3 * query_idx + 0] = 0.0f;
//     normals[3 * query_idx + 1] = 0.0f;
//     normals[3 * query_idx + 2] = 0.0f;

//     float3 query_point =       
//     {
//         query_points[3 * query_idx + 0],
//         query_points[3 * query_idx + 1],
//         query_points[3 * query_idx + 2]
//     };
    
//     query_knn(nodes, points, sorted_indices, root_index, &query_point, queue);
//     __syncwarp(); // synchronize the warp before the write operation

//     __syncthreads();

//     // http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
//     unsigned int n = queue.size();
//     const float nf = (float) n;


//     if(n < 3)
//     {
//         // Not enough neighbors
//         normals[3 * query_idx + 0] = 0.0f;
//         normals[3 * query_idx + 1] = 0.0f;
//         normals[3 * query_idx + 2] = 0.0f;
//         return; 
//     }


//     // Get the centroid
//     float3 sum = {0.0f, 0.0f, 0.0f};

//     for(int i = 0; i < n; i++)
//     {
//         auto k = queue[i];
//         // Coordinates of point which is the i-th neighbor
//         sum.x += points[ 3 * k.id + 0];
//         sum.y += points[ 3 * k.id + 1];
//         sum.z += points[ 3 * k.id + 2];
//     }

//     sum.x /= nf; // x,y,z coordinates of centroid
//     sum.y /= nf;
//     sum.z /= nf;

//     // Calculate the covariance matrix
//     float xx = 0.0f; float xy = 0.0f; float xz = 0.0f;
//     float yy = 0.0f; float yz = 0.0f;
//     float zz = 0.0f;

//     for(int i = 0; i < n; i++)
//     {
//         auto k = queue[i];
//         float3 r =
//         {
//             points[ 3 * k.id + 0] - sum.x,
//             points[ 3 * k.id + 1] - sum.y,
//             points[ 3 * k.id + 2] - sum.z
//         };

//         xx += r.x * r.x;
//         xy += r.x * r.y;
//         xz += r.x * r.z;
//         yy += r.y * r.y;
//         yz += r.y * r.z;
//         zz += r.z * r.z;
        
//     }

//     xx /= nf;
//     xy /= nf;
//     xz /= nf;
//     yy /= nf;
//     yz /= nf;
//     zz /= nf;
    
//     float3 weighted_dir = {0.0f, 0.0f, 0.0f};
//     float3 axis_dir;

//     // For x
//     float det_x = yy*zz - yz*yz;

//     axis_dir.x = det_x;
//     axis_dir.y = xz*yz - xy*zz;
//     axis_dir.z = xy*yz - xz*yy;

//     float weight = det_x * det_x;

//     if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0f)
//     {
//         weight *= -1;
//     }
    
//     weighted_dir.x += axis_dir.x * weight;
//     weighted_dir.y += axis_dir.y * weight;
//     weighted_dir.z += axis_dir.z * weight;


//     // For y
//     float det_y = xx*zz - xz*xz;

//     axis_dir.x = xz*yz - xy*zz;
//     axis_dir.y = det_y;
//     axis_dir.z = xy*xz - yz*xx;

//     weight = det_y * det_y;

//     if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0f)
//     {
//         weight *= -1;
//     }
    
//     weighted_dir.x += axis_dir.x * weight;
//     weighted_dir.y += axis_dir.y * weight;
//     weighted_dir.z += axis_dir.z * weight;


//     // For z
//     float det_z = xx*yy - xy*xy;

//     axis_dir.x = xy*yz - xz*yy;
//     axis_dir.y = xy*xz - yz*xx;
//     axis_dir.z = det_z;

//     weight = det_z * det_z;

//     if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0f)
//     {
//         weight *= -1;
//     }
    
//     weighted_dir.x += axis_dir.x * weight;
//     weighted_dir.y += axis_dir.y * weight;
//     weighted_dir.z += axis_dir.z * weight;

//     // Create the normal
//     float3 normal = weighted_dir;

//     // Normalize normal
//     float mag = sqrt((normal.x * normal.x) + (normal.y * normal.y) + (normal.z * normal.z));
   
//     normal.x /= mag;
//     normal.y /= mag;
//     normal.z /= mag;

//     // Check if the normals need to be flipped
//     float vertex_x = query_points[3 * query_idx + 0];
//     float vertex_y = query_points[3 * query_idx + 1];
//     float vertex_z = query_points[3 * query_idx + 2];

//     // flip the normals
//     float x_dir = flip_x - vertex_x;
//     float y_dir = flip_y - vertex_y;
//     float z_dir = flip_z - vertex_z;

//     float scalar = x_dir * normal.x + y_dir * normal.y + z_dir * normal.z;

//     if(scalar < 0)
//     {
//         normal.x = -normal.x;
//         normal.y = -normal.y;
//         normal.z = -normal.z;
//     }

//     // Set normals to zero if nan or inf values occur
//     if(!(scalar <= 0 || scalar >= 0) || isinf(scalar))
//     {
//         normal.x = 0.0f;
//         normal.y = 0.0f;
//         normal.z = 0.0f;
//     }
   
//     // Set the normal in the normal array
//     normals[3 * query_idx + 0] = normal.x;
//     normals[3 * query_idx + 1] = normal.y;
//     normals[3 * query_idx + 2] = normal.z;

//     return;
// }

// } // namespace lvr2


#include "query_knn.cuh"

using namespace lvr2;
using namespace lbvh;

namespace lvr2
{
/**
 * @brief   A Cuda kernel that performs a kNN search on the LBVH and
 *          calculates the surface normals by doing an approximated
 *          iterative PCA
 * 
 * @param nodes             Nodes of the LBVH
 * @param points            Points of the dataset
 * @param sorted_indices    Sorted indices of the points
 * @param root_index        Index of the LBVH root node
 * @param max_radius        Maximum radius for radius search
 * @param query_points      Query points for which the normals are calculated
 * @param sorted_queries    Sorted indices of the query points
 * @param num_queries       Number of queries
 * @param normals           Stores the calculated normals
 */
extern "C" __global__ void knn_normals_kernel(
    const BVHNode *nodes,
    const float* __restrict__ points,         
    const unsigned int* __restrict__ sorted_indices,
    const unsigned int root_index,
    const float max_radius,
    const float* __restrict__ query_points,    
    const unsigned int* __restrict__ sorted_queries,
    const unsigned int num_queries, 
    float* normals
)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (tid >= num_queries)
    {
        return;
    }

    double flip_x=1000000.0; 
    double flip_y=1000000.0; 
    double flip_z=1000000.0;

    StaticPriorityQueue<float, K> queue(max_radius);
    unsigned int query_idx = sorted_queries[tid];

    // TODO Just for testing
    // if(query_idx != 59394) 
    // {
    //     return;
    // }

    float3 query_point =       
    {
        query_points[3 * query_idx + 0],
        query_points[3 * query_idx + 1],
        query_points[3 * query_idx + 2]
    };
    
    query_knn(nodes, points, sorted_indices, root_index, &query_point, queue);
    __syncwarp(); // synchronize the warp before the write operation

    // __syncthreads();

    // http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html

    // We only consider n - 1 neighbors, since the first found neighbor 
    // is the query point itself
    unsigned int n = queue.size();      // only used in for loops
    const double nf = (double) n - 1;   // used for calculation, therefore -1

    if(n < 3)
    {
        // Not enough neighbors
        normals[3 * query_idx + 0] = 0.0f;
        normals[3 * query_idx + 1] = 0.0f;
        normals[3 * query_idx + 2] = 0.0f;
        return; 
    }


    // Get the centroid
    double3 sum = {0.0, 0.0, 0.0};

    for(int i = 1; i < n; i++)
    {
        auto k = queue[i];
        // Coordinates of point which is the i-th neighbor
        sum.x += (double) points[ 3 * k.id + 0] / nf;
        sum.y += (double) points[ 3 * k.id + 1] / nf;
        sum.z += (double) points[ 3 * k.id + 2] / nf;
        // TODO
        // normals[3 * query_idx + 0 + 3 * (i)] = points[ 3 * k.id + 0];
        // normals[3 * query_idx + 1 + 3 * (i)] = points[ 3 * k.id + 1];
        // normals[3 * query_idx + 2 + 3 * (i)] = points[ 3 * k.id + 2];

        // printf("Point %d: (%f %f %f)\n", k.id, points[ 3 * k.id + 0], points[ 3 * k.id + 1], points[ 3 * k.id + 2]);
    }

    // Calculate the covariance matrix
    double xx = 0.0f; double xy = 0.0f; double xz = 0.0f;
    double yy = 0.0f; double yz = 0.0f;
    double zz = 0.0f;

    for(int i = 1; i < n; i++)
    {
        auto k = queue[i];
        double3 r =
        {
            (double) points[ 3 * k.id + 0] - sum.x,     
            (double) points[ 3 * k.id + 1] - sum.y,    
            (double) points[ 3 * k.id + 2] - sum.z      
        };
        xx += r.x * r.x / nf;  
        xy += r.x * r.y / nf;   
        xz += r.x * r.z / nf;   
        yy += r.y * r.y / nf;   
        yz += r.y * r.z / nf;   
        zz += r.z * r.z / nf;   
    }
    
    double3 weighted_dir = {0.0, 0.0, 0.0};
    double3 axis_dir;

    // For x
    double det_x = yy*zz - yz*yz;   // 0

    axis_dir.x = det_x;
    axis_dir.y = xz*yz - xy*zz;
    axis_dir.z = xy*yz - xz*yy;

    double weight = det_x * det_x;

    if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0)
    {
        weight *= -1;
    }
    
    weighted_dir.x += axis_dir.x * weight;
    weighted_dir.y += axis_dir.y * weight;
    weighted_dir.z += axis_dir.z * weight;


    // For y
    double det_y = xx*zz - xz*xz;   // 0

    axis_dir.x = xz*yz - xy*zz;
    axis_dir.y = det_y;
    axis_dir.z = xy*xz - yz*xx;

    weight = det_y * det_y;

    if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0)
    {
        weight *= -1;
    }
    
    weighted_dir.x += axis_dir.x * weight;
    weighted_dir.y += axis_dir.y * weight;
    weighted_dir.z += axis_dir.z * weight;


    // For z
    double det_z = xx*yy - xy*xy;   // 0

    axis_dir.x = xy*yz - xz*yy;
    axis_dir.y = xy*xz - yz*xx;
    axis_dir.z = det_z;

    weight = det_z * det_z;

    if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0)
    {
        weight *= -1;
    }
    
    weighted_dir.x += axis_dir.x * weight;
    weighted_dir.y += axis_dir.y * weight;
    weighted_dir.z += axis_dir.z * weight;

    // Create the normal
    double3 normal = weighted_dir;  // (0,0,0)

    // Normalize normal
    double mag = sqrt((normal.x * normal.x) + (normal.y * normal.y) + (normal.z * normal.z));   // 0
   
    normal.x /= mag;
    normal.y /= mag;
    normal.z /= mag;

    // Check if the normals need to be flipped
    double vertex_x = query_points[3 * query_idx + 0];
    double vertex_y = query_points[3 * query_idx + 1];
    double vertex_z = query_points[3 * query_idx + 2];

    // flip the normals
    double x_dir = flip_x - vertex_x;   // 1e+06
    double y_dir = flip_y - vertex_y;   // 1.00001e+06
    double z_dir = flip_z - vertex_z;   // 999997

    double scalar = x_dir * normal.x + y_dir * normal.y + z_dir * normal.z; // -nan

    // printf("Scalar: %f\n", scalar);

    // Set normals to zero if nan or inf values occur
    if(!(scalar <= 0 || scalar >= 0) || isinf(scalar))
    {
        normal.x = 0.0f;
        normal.y = 0.0f;
        normal.z = 0.0f;
    }

    if(scalar < 0)
    {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }

    // printf("Normal: (%f, %f, %f) \n", normal.x, normal.y, normal.z);

    // Set the normal in the normal array
    normals[3 * query_idx + 0] = (float) normal.x;
    normals[3 * query_idx + 1] = (float) normal.y;
    normals[3 * query_idx + 2] = (float) normal.z;

    return;
}

} // namespace lvr2
