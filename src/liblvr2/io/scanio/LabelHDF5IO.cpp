#include "lvr2/io/scanio/LabelHDF5IO.hpp"

namespace lvr2
{

LabelHDF5IO::LabelHDF5IO(HDF5KernelPtr kernel, LabelHDF5SchemaPtr schema)
    : m_kernel(kernel), m_schema(schema)
{

}

void LabelHDF5IO::saveLabelScanProject(LabeledScanProjectEditMarkPtr project)
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::LabelScanProjectIO>;

    MyScanProjectIO io(m_kernel, m_schema);
    io.saveLabelScanProject(project);
}

LabeledScanProjectEditMarkPtr LabelHDF5IO::loadScanProject()
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::LabelScanProjectIO>;

    MyScanProjectIO io(m_kernel, m_schema);
    return io.loadLabelScanProject(); 
}

} // namespace lvr2

