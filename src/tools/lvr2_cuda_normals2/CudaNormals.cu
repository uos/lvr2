// #include <boost/filesystem.hpp>
// #include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// #include "lvr2/io/ModelFactory.hpp"
// #include "lvr2/util/Timestamp.hpp"
// #include "lvr2/util/IOUtils.hpp"
// #include "Options.hpp"
#include "CudaNormals.cuh"

__global__
void initNormals_kernel(float* normals, size_t num_points)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_points)
    {
        normals[i * 3 + 0] = 1.0;
        normals[i * 3 + 1] = 0.0;
        normals[i * 3 + 2] = 0.0;
    }    
}

void initNormals(float* h_normals, size_t num_points)
{
    int size = num_points * 3;

    float* d_normals;
    cudaMalloc(&d_normals, size);

    // Initialize the normals
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    // int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    printf("%d %d \n", threadsPerBlock, blocksPerGrid);

    initNormals_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_normals, num_points);
    // initNormals_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_normals, size);

    cudaDeviceSynchronize();

    // Copy the normals back to host
    cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    cudaFree(d_normals);
}


// void setNormals(int argc, char** argv)
// {
//     cuda_normals_2::Options opt(argc, argv);
//     cout << opt << endl;

//     // Get the model
//     ModelPtr model = ModelFactory::readModel(opt.inputFile());

//     // Get the points
//     PointBufferPtr pbuffer = model->m_pointCloud;
//     size_t num_points = model->m_pointCloud->numPoints();

//     floatArr points = pbuffer->getPointArray();
    
//     float* points_raw = &points[0];
    
//     // floatArr normals(new float[num_points * 3]);
//     // Create normals arrays and copy to device
//     int size = num_points * 3;
//     float* h_normals = (float*)malloc(size);

//     float* d_normals;
//     cudaMalloc(&d_normals, size);

//     cudaMemcpy(d_normals, h_normals, size, cudaMemcpyHostToDevice);

//     // Initialize the normals
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

//     initNormals<<<blocksPerGrid, threadsPerBlock>>>(d_normals, num_points);

//     // Copy the normals back to host
//     cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

//     // Write the normals into the model
//     floatArr normals(new float[size]);

//     for(int i = 0; i < size; i++)
//     {
//         normals[i] = h_normals[i];
//     }

//     pbuffer->setNormalArray(normals, num_points);

//     ModelFactory::saveModel(model, "test.ply");

//     // Free memory
//     cudaFree(d_normals);
//     free(h_normals);
// }
