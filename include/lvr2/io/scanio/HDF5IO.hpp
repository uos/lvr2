#ifndef HDF5_DESCRIPTION_IO_HPP
#define HDF5_DESCRIPTION_IO_HPP

#include "lvr2/io/kernels/HDF5Kernel.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <memory>

namespace lvr2
{
namespace scanio
{

using HDF5IOBase = lvr2::baseio::FeatureBuild<ScanProjectIO>;

class HDF5IO : public HDF5IOBase
{
public:
    HDF5IO(HDF5KernelPtr kernel, HDF5SchemaPtr schema, bool load_data = false)
    : HDF5IOBase(kernel, schema, load_data)
    { }
};

using HDF5IOPtr = std::shared_ptr<HDF5IO>;

} // namepsace scanio
} // namespace lvr2

#endif
