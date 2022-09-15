#include "MortonCodes.cuh"

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
unsigned long long int expandBits(unsigned long long int v)
{
    v = (v * 0x000100000001u) & 0xFFFF00000000FFFFu;
    v = (v * 0x000000010001u) & 0x00FF0000FF0000FFu;
    v = (v * 0x000000000101u) & 0xF00F00F00F00F00Fu;
    v = (v * 0x000000000011u) & 0x30C30C30C30C30C3u;
    v = (v * 0x000000000005u) & 0x9249249249249249u;
        
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
unsigned long long int morton3D(float x, float y, float z)
{
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);

    unsigned long long int xx = expandBits((unsigned int)x);
    unsigned long long int yy = expandBits((unsigned int)y);
    unsigned long long int zz = expandBits((unsigned int)z);

    return xx * 4 + yy * 2 + zz;
}

// Get the extent of the points 
// (minimum and maximum values in each dimension)
Extent getExtent(float* h_points, size_t num_points)
{
    float min_x = INT_MAX;
    float min_y = INT_MAX; 
    float min_z = INT_MAX;

    float max_x = INT_MIN; 
    float max_y = INT_MIN; 
    float max_z = INT_MIN;

    for(int i = 0; i < num_points; i += 3)
    {
        if(h_points[i + 0] < min_x)
        {
            min_x = h_points[i + 0];
        }

        if(h_points[i + 1] < min_y)
        {
            min_y = h_points[i + 1];
        }

        if(h_points[i + 2] < min_z)
        {
            min_z = h_points[i + 2];
        }

        if(h_points[i + 0] > max_x)
        {
            max_x = h_points[i + 0];
        }

        if(h_points[i + 1] > max_y)
        {
            max_y = h_points[i + 1];
        }

        if(h_points[i + 2] > max_z)
        {
            max_z = h_points[i + 2];
        }

    }
    Extent extent;
    extent.min.x = min_x;
    extent.min.y = min_y;
    extent.min.z = min_z;
    
    extent.max.x = max_x;
    extent.max.y = max_y;
    extent.max.z = max_z;
    
    return extent;
}

__global__ 
void mortonCode(unsigned long long int* mortonCodes, float* points, size_t num_points, Extent extent)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= num_points)
    {
        return;
    }
    
    // Get xyz coordinates of the point
    float p_x = points[idx + 0];
    float p_y = points[idx + 1];
    float p_z = points[idx + 2];

    // Scale to [0, 1] (unit cube)
    p_x -= extent.min.x;
    p_y -= extent.min.y;
    p_z -= extent.min.z;

    p_x /= (extent.max.x - extent.min.x);
    p_y /= (extent.max.y - extent.min.y);
    p_z /= (extent.max.z - extent.min.z);

    if(idx == 0) 
    {
        printf("x: %f \n", p_x);

        printf("y: %f \n", p_y);

        printf("z: %f \n", p_z);
    }

    mortonCodes[idx] = morton3D(p_x, p_y, p_z);

    return;
}


void getMortonCodes(unsigned long long int* h_mortonCodes, float* h_points, size_t num_points)
{
    int size_points = num_points * 3 * sizeof(float);
    int size_morton = num_points * sizeof(unsigned long long int);
    // Get the extent of the point cloud
    Extent extent = getExtent(h_points, num_points); 

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    float* d_points;
    cudaMalloc(&d_points, size_points);
    cudaMemcpy(d_points, h_points, size_points, cudaMemcpyHostToDevice);

    unsigned long long int* d_mortonCodes;
    cudaMalloc(&d_mortonCodes, size_morton);
    cudaMemcpy(d_mortonCodes, h_mortonCodes, size_morton, cudaMemcpyHostToDevice);

    mortonCode<<<blocksPerGrid, threadsPerBlock>>>(d_mortonCodes, d_points, num_points, extent);

    cudaMemcpy(h_mortonCodes, d_mortonCodes, size_morton, cudaMemcpyDeviceToHost);

    return;
}