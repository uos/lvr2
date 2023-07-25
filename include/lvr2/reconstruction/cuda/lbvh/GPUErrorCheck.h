#pragma once
#ifndef LVR2_GPU_ERROR_CHECK_H
#define LVR2_GPU_ERROR_CHECK_H
#ifndef NDEBUG

#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#else 
#define gpuErrchk(ans) {ans;} 
#endif
#endif // LVR2_GPU_ERROR_CHECK_H