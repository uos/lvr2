#include "lvr2/io/descriptions/HDF5IO.hpp"

namespace lvr2
{
namespace descriptions
{

HDF5IO::HDF5IO(HDF5KernelPtr kernel, HDF5SchemaPtr schema)
    : m_kernel(kernel), m_schema(schema)
{

}

void HDF5IO::saveScanProject(ScanProjectPtr project)
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::LabelScanProjectIO>;

    LabeledScanProjectEditMarkPtr labelProject;
    labelProject->editMarkProject->project = project;
    MyScanProjectIO io(m_kernel, m_schema);
    io.saveLabelScanProject(labelProject);
}

ScanProjectPtr HDF5IO::loadScanProject()
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::LabelScanProjectIO>;

    MyScanProjectIO io(m_kernel, m_schema);
    LabeledScanProjectEditMarkPtr ptr = io.loadLabelScanProject();
    return ptr->editMarkProject->project;
}
} // namespace descriptions
} // namespace lvr2

