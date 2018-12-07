#include <lvr2/io/ScanDataManager.hpp>

namespace lvr2
{

ScanDataManager::ScanDataManager(std::string filename) : m_io(filename)
{
}

void ScanDataManager::loadPointCloudData(ScanData& sd)
{
    if (!sd.m_pointsLoaded)
    {
        sd = m_io.getSingleScanData(sd.m_scanDataRoot, sd.m_positionNumber, true);
    }
}

std::vector<std::string> ScanDataManager::getScanDataGroups()
{
    return m_io.getScanDataGroups();
}

std::vector<ScanData> ScanDataManager::getScanData(std::string scanDataGroup)
{
    return m_io.getScanData(scanDataGroup, false);
}

} // namespace lvr2
