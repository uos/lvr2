#include <lvr2/io/ScanDataManager.hpp>

namespace lvr2
{

ScanDataManager::ScanDataManager(std::string filename) : m_io(filename)
{
    m_scanData = m_io.getRawScanData(false);
}

ScanData ScanDataManager::loadPointCloudData(size_t idx)
{
    if (!m_scanData[idx].m_pointsLoaded)
    {
        m_scanData[idx] = m_io.getRawScanData(m_scanData[idx].m_positionNumber, true);
    }

    return m_scanData[idx];
}

std::vector<ScanData>& ScanDataManager::getScanData()
{
    return m_scanData;
}

} // namespace lvr2
