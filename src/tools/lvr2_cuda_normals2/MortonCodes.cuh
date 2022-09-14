#ifndef MORTONCODES_CUH
#define MOTONCODES_CUH

#include <stdio.h>
#include <cuda_runtime.h>

struct Extent
{
    float3 min;
    float3 max;
};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v);

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z);


void getMortonCodes(float* h_points, size_t num_points);

#endif // MORTONCODES_CUH