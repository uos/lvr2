#ifndef LABELHDF5_DESCRIPTION_IO_HPP
#define LABELHDF5_DESCRIPTION_IO_HPP

#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/ScanProjectSchema.hpp"
#include "lvr2/io/descriptions/FeatureBase.hpp"
#include "lvr2/io/descriptions/LabelScanProjectIO.hpp"
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
