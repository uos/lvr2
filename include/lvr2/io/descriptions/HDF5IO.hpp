#ifndef HDF5_DESCRIPTION_IO_HPP
#define HDF5_DESCRIPTION_IO_HPP

#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/FeatureBase.hpp"
#include "lvr2/io/descriptions/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{
namespace descriptions
{

using HDF5IOBase = FeatureBuild<ScanProjectIO>;

class HDF5IO : public HDF5IOBase
{
public:
    HDF5IO(HDF5KernelPtr kernel, HDF5SchemaPtr schema)
    : HDF5IOBase(kernel, schema)
    { }
};

} // namepsace descriptions
} // namespace lvr2

#endif
