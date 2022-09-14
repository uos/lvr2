#include "MortonCodes.cuh"

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z)
{
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}


void getMortonCodes(float* h_points, size_t num_points)
{
    // Get the extent of the points 
    // (minimum and maximum values in each dimension)
    float min_x, min_y, min_z = INT_MAX;
    float max_x, max_y, max_z = INT_MIN;

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
    
    printf("Min: %f, %f, %f \n", extent.min.x, extent.min.y, extent.min.z);
    printf("Max: %f, %f, %f \n", extent.max.x, extent.max.y, extent.max.z); 
}
