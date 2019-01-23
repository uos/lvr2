#include <lvr2/io/ScanDataManager.hpp>

namespace lvr2
{

ScanDataManager::ScanDataManager(std::string filename) : m_io(filename)
{
}

void ScanDataManager::loadPointCloudData(ScanData& sd, bool preview)
{
    if ((!sd.m_pointsLoaded && !preview) || ( sd.m_pointsLoaded && preview))
    {
        sd = m_io.getSingleRawScanData(sd.m_positionNumber, !preview);
    }
}

std::vector<ScanData> ScanDataManager::getScanData()
{
    return m_io.getRawScanData(false);
}

} // namespace lvr2
