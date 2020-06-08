#include "lvr2/io/descriptions/DirectoryIO.hpp"

namespace lvr2
{

void DirectoryIO::saveScanProject(ScanProjectPtr project)
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::ScanProjectIO>;

    MyScanProjectIO io(m_kernel, m_schema);
    io.saveScanProject(project);
}

ScanProjectPtr DirectoryIO::loadScanProject()
{
    using BaseScanProjectIO = lvr2::FeatureBase<>;
    using MyScanProjectIO = BaseScanProjectIO::AddFeatures<lvr2::ScanProjectIO>;

    MyScanProjectIO io(m_kernel, m_schema);
    ScanProjectPtr ptr = io.loadScanProject();
    return ptr;
}

} // namespace lvr2

