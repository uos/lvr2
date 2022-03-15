#ifndef LABELHDF5_DESCRIPTION_IO_HPP
#define LABELHDF5_DESCRIPTION_IO_HPP

#include "lvr2/io/kernels/HDF5Kernel.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/io/scanio/FeatureBase.hpp"
#include "lvr2/io/scanio/LabelScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{
class LabelHDF5IO
{
public:

    LabelHDF5IO(HDF5KernelPtr kernel, LabelHDF5SchemaPtr schema);

    void saveLabelScanProject(LabeledScanProjectEditMarkPtr project);
    LabeledScanProjectEditMarkPtr loadScanProject();

private:
    HDF5KernelPtr   m_kernel;
    LabelHDF5SchemaPtr   m_schema;
};
} // namespace lvr2

#endif
