#include <boost/filesystem.hpp>
#include <iostream>

#include <cuda_runtime.h>
#include <driver_types.h>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/util/IOUtils.hpp"
#include "Options.hpp"

#include "CudaNormals.cu"

using namespace lvr2;

int main(int argc, char** argv)
{
    setNormals(argc, argv);

    return 0;
}
