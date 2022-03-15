#pragma once

#include "lvr2/io/kernels/DirectoryKernel.hpp"
#include <lvr2/io/schema/MeshSchemaDirectory.hpp>
#include <lvr2/io/meshio/MeshIO.hpp>
#include <lvr2/io/scanio/FeatureBase.hpp>

namespace lvr2
{
namespace meshio
{

using DirectoryIOBase = FeatureBuild<MeshIO, MeshSchemaDirectoryPtr>;

class DirectoryIO : public DirectoryIOBase
{
public:
    DirectoryIO(DirectoryKernelPtr kernel, MeshSchemaDirectoryPtr schema) 
    : DirectoryIOBase(kernel, schema) 
    {}
};

using DirectoryIOPtr = std::shared_ptr<DirectoryIO>;

} // namespace meshio
} // namespace lvr2