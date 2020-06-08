#ifndef HDF5_IO_HPP
#define HDF5_IO_HPP

#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/FeatureBase.hpp"
#include "lvr2/io/descriptions/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

class HDF5IO
{
public:
    HDF5IO(HDF5KernelPtr kernel, HDF5SchemaPtr schema);

    void saveScanProject(ScanProjectPtr project);
    ScanProjectPtr loadScanProject();

private:
    HDF5KernelPtr   m_kernel;
    HDF5SchemaPtr   m_schema;
};

} // namespace lvr2

#endif