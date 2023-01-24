#include "query_knn.cuh"

using namespace lbvh;

extern "C" __global__ void knn_normals_kernel(
    const BVHNode *nodes,
    const float* __restrict__ points,         
    const unsigned int* __restrict__ sorted_indices,
    const unsigned int root_index,
    const float max_radius,
    const float* __restrict__ query_points,    
    const unsigned int* __restrict__ sorted_queries,
    const unsigned int num_queries, 
    // custom parameters
    float* normals
    //float flip_x=1000000.0, float flip_y=1000000.0, float flip_z=1000000.0
)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_queries)
    {
        return;
    }

    float flip_x=1000000.0; 
    float flip_y=1000000.0; 
    float flip_z=1000000.0;

    StaticPriorityQueue<float, K> queue(max_radius);
    unsigned int query_idx = sorted_queries[tid];

    normals[3 * query_idx + 0] = 0.0f;
    normals[3 * query_idx + 1] = 0.0f;
    normals[3 * query_idx + 2] = 0.0f;

    // unsigned int indices_out[K];
    // float distances_out[K];
    // unsigned int n_neighbors_out[1];

    // unsigned int index_out;
    // float distance_out;
    // unsigned int n_neighbors_out;

    // for(int i = 0; i < K; i++)
    // {
    //     indices_out[K * query_idx + i] = UINT_MAX;
    //     distances_out[K * query_idx + i] = FLT_MAX;
    // }

    float3 query_point =       
    {
        query_points[3 * query_idx + 0],
        query_points[3 * query_idx + 1],
        query_points[3 * query_idx + 2]
    };
    // query_knn(nodes, points, sorted_indices, root_index, &query_points[query_idx], queue);
    query_knn(nodes, points, sorted_indices, root_index, &query_point, queue);
    __syncwarp(); // synchronize the warp before the write operation
    // write back the results at the correct position
    // queue.write_results(&indices_out[0], &distances_out[0], &n_neighbors_out[0]);
    // queue.write_results(&index_out, &distance_out, &n_neighbors_out);

    // http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
    __syncthreads();

    unsigned int n = queue.size();
    const float nf = (float) n;


    if(n < 3)
    {
        // TODO tid or query_idx?
        normals[3 * query_idx + 0] = 0.0f;
        normals[3 * query_idx + 1] = 0.0f;
        normals[3 * query_idx + 2] = 0.0f;
        return; // Not enough neighbors
    }


    // Get the centroid
    float3 sum = {0.0f, 0.0f, 0.0f};

    for(int i = 0; i < n; i++)
    {
        // TODO Indexing?
        auto k = queue[query_idx * K + i];
        // Coordinates of point which is the i-th neighbor
        sum.x += points[ 3 * k.id + 0];
        sum.y += points[ 3 * k.id + 1];
        sum.z += points[ 3 * k.id + 2];
    }

    sum.x /= nf; // x,y,z coordinates of centroid
    sum.y /= nf;
    sum.z /= nf;

    // Calculate the covariance matrix
    float xx = 0.0f; float xy = 0.0f; float xz = 0.0f;
    float yy = 0.0f; float yz = 0.0f;
    float zz = 0.0f;

    for(int i = 0; i < n; i++)
    {
        // TODO Indexing?
        auto k = queue[query_idx * K + i];
        float3 r =
        {
            points[ 3 * k.id + 0] - sum.x,
            points[ 3 * k.id + 1] - sum.y,
            points[ 3 * k.id + 2] - sum.z
        };

        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
        
    }

    xx /= nf;
    xy /= nf;
    xz /= nf;
    yy /= nf;
    yz /= nf;
    zz /= nf;
    
    float3 weighted_dir = {0.0f, 0.0f, 0.0f};
    float3 axis_dir;

    // For x
    float det_x = yy*zz - yz*yz;

    axis_dir.x = det_x;
    axis_dir.y = xz*yz - xy*zz;
    axis_dir.z = xy*yz - xz*yy;

    float weight = det_x * det_x;

    if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0f)
    {
        weight *= -1;
    }
    
    weighted_dir.x += axis_dir.x * weight;
    weighted_dir.y += axis_dir.y * weight;
    weighted_dir.z += axis_dir.z * weight;


    // For y
    float det_y = xx*zz - xz*xz;

    axis_dir.x = xz*yz - xy*zz;
    axis_dir.y = det_y;
    axis_dir.z = xy*xz - yz*xx;

    weight = det_y * det_y;

    if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0f)
    {
        weight *= -1;
    }
    
    weighted_dir.x += axis_dir.x * weight;
    weighted_dir.y += axis_dir.y * weight;
    weighted_dir.z += axis_dir.z * weight;


    // For z
    float det_z = xx*yy - xy*xy;

    axis_dir.x = xy*yz - xz*yy;
    axis_dir.y = xy*xz - yz*xx;
    axis_dir.z = det_z;

    weight = det_z * det_z;

    if(weighted_dir.x * axis_dir.x + weighted_dir.y * weighted_dir.y + weighted_dir.z * weighted_dir.z < 0.0f)
    {
        weight *= -1;
    }
    
    weighted_dir.x += axis_dir.x * weight;
    weighted_dir.y += axis_dir.y * weight;
    weighted_dir.z += axis_dir.z * weight;

    // Create the normal
    float3 normal = weighted_dir;

    // Normalize normal
    float mag = sqrt((normal.x * normal.x) + (normal.y * normal.y) + (normal.z * normal.z));
   
    normal.x /= mag;
    normal.y /= mag;
    normal.z /= mag;

    // Check if the normals need to be flipped
    float vertex_x = query_points[3 * query_idx + 0];
    float vertex_y = query_points[3 * query_idx + 1];
    float vertex_z = query_points[3 * query_idx + 2];

    // flip the normals
    float x_dir = flip_x - vertex_x;
    float y_dir = flip_y - vertex_y;
    float z_dir = flip_z - vertex_z;

    float scalar = x_dir * normal.x + y_dir * normal.y + z_dir * normal.z;

    // TODO < or > ?
    if(scalar < 0)
    {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }

    // TODO Set normals to zero if nan or inf values occur
    // Done?
    if(!(scalar <= 0 || scalar >= 0) || isinf(scalar))
    {
        normal.x = 0.0f;
        normal.y = 0.0f;
        normal.z = 0.0f;
    }

    // float x = normal.x;
    // float y = normal.y;
    // float z = normal.z;

    // TODO Is the indexing here correct?
    // Set the normal in the normal array
    // TODO tid or query_idx?
    normals[3 * query_idx + 0] = normal.x;
    normals[3 * query_idx + 1] = normal.y;
    normals[3 * query_idx + 2] = normal.z;

    // normals[3 * query_idx + 0] = 0.0f;
    // normals[3 * query_idx + 1] = 0.0f;
    // normals[3 * query_idx + 2] = 0.0f;

    return;
}