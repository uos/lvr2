#include "normals_kernel.cuh"

#include <stdio.h>

using namespace lbvh;

__global__ void lbvh::calculate_normals_kernel(float* points, 
    float* queries, size_t num_queries, 
    unsigned int* n_neighbors_out, unsigned int* indices_out, 
    unsigned int* neigh_sum,
    float* normals)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= num_queries)
    {
        return;
    }

    // http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html

    int n = n_neighbors_out[tid];

    if(n < 3)
    {
        return; // Not enough neighbors
    }

    // Get the centroid
    float3 sum = {0.0f, 0.0f, 0.0f};

    for(int i = 0; i < n; i++)
    {
        sum.x += points[ 3 * indices_out[ neigh_sum[tid] + i] + 0];
        sum.y += points[ 3 * indices_out[ neigh_sum[tid] + i] + 1];
        sum.z += points[ 3 * indices_out[ neigh_sum[tid] + i] + 2];
    }

    sum.x /= n; // x,y,z coordinates of centroid
    sum.y /= n;
    sum.z /= n;

    // Calculate the covariance matrix
    float xx = 0.0f; float xy = 0.0f; float xz = 0.0f;
    float yy = 0.0f; float yz = 0.0f;
    float zz = 0.0f;

    for(int i = 0; i < n; i++)
    {
        float3 r = 
        {
            points[ 3 * indices_out[ neigh_sum[tid] + i] + 0] - sum.x,
            points[ 3 * indices_out[ neigh_sum[tid] + i] + 1] - sum.y,
            points[ 3 * indices_out[ neigh_sum[tid] + i] + 2] - sum.z
        };

        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
        
    }

    xx /= n;
    xy /= n;
    xz /= n;
    yy /= n;
    yz /= n;
    zz /= n;

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

    // Set the normal in the normal array
    normals[3 * tid + 0] = normal.x;
    normals[3 * tid + 1] = normal.y;
    normals[3 * tid + 2] = normal.z;


    // if(tid == num_queries - 1)
    //     printf("All working fine. \n");


    // if(tid == 0)
    // {   

    //     printf("Checking kernel... \n");
    //     printf("%f \n", points[0]);
    //     printf("%d \n", num_normals);
    //     printf("%d \n", n_neighbors_out[0]);
    //     printf("%d \n", indices_out[0]);
    //     printf("%d \n", neigh_sum[0]);
    //     printf("%f \n", normals[0]);
    //     printf("Kernel working fine \n");
    // }
}