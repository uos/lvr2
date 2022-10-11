#include "kernels_device.cuh"
#include "morton_code.cuh"

using namespace lbvh;

__global__
void lbvh::morton_code_kernel(unsigned long long int* d_mortonCodes, 
                    float* d_points, 
                    int num_points, 
                    AABB extent)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float3* points = reinterpret_cast<float3*>(d_points);

    if(idx >= num_points)
    {
        return;
    }
    
    // Get xyz coordinates of the point
    float p_x = points[idx].x;
    float p_y = points[idx].y;
    float p_z = points[idx].z;
    
    // Scale to [0, 1] (unit cube)
    p_x -= extent.min.x;
    p_y -= extent.min.y;
    p_z -= extent.min.z;

    p_x /= (extent.max.x - extent.min.x);
    p_y /= (extent.max.y - extent.min.y);
    p_z /= (extent.max.z - extent.min.z);

    float3 p;
    p.x = p_x;
    p.y = p_y;
    p.z = p_z;

    d_mortonCodes[idx] = morton_code(p, 1024.0f);

    return;

}
