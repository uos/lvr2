#include "lvr2/io/ScanDataManager.hpp"

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

std::vector<std::vector<CameraData> > ScanDataManager::getCameraData()
{
    return m_io.getRawCamData(false);
}

cv::Mat ScanDataManager::loadImageData(int scan_id, int cam_id)
{
    CameraData ret = m_io.getSingleRawCamData(scan_id, cam_id, true);
    return ret.image;
}


} // namespace lvr2
