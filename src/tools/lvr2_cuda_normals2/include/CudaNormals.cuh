#ifndef CUDA_NORMALS_CUH
#define CUDA_NORMALS_CUH

// #include <boost/filesystem.hpp>
// #include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// #include "lvr2/io/ModelFactory.hpp"
// #include "lvr2/util/Timestamp.hpp"
// #include "lvr2/util/IOUtils.hpp"
// #include "Options.hpp"

void initNormals(float* h_normals, size_t num_points);
void initNormals2(float* h_normals, size_t num_points);

#endif // CUDA_NORMALS_CUH