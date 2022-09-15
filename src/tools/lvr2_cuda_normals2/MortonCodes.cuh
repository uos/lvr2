#ifndef MORTONCODES_CUH
#define MOTONCODES_CUH

#include <stdio.h>
#include <cuda_runtime.h>

struct Extent
{
    float3 min;
    float3 max;
};

void getMortonCodes(unsigned long long int* mortonCodes, float* h_points, size_t num_points);

// Calculates the extent (minimun and maximum values in each dimension)
Extent getExtent(float* h_points, size_t num_points);

#endif // MORTONCODES_CUH