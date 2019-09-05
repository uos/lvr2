#include "lvr2/io/ScanDataManager.hpp"

namespace lvr2
{

ScanDataManager::ScanDataManager(std::string filename) : m_io(filename)
{
    
}

void ScanDataManager::loadPointCloudData(ScanPtr& sd, bool preview)
{
    if ((!sd->m_pointsLoaded && !preview) || ( sd->m_pointsLoaded && preview))
    {
        sd = m_io.getSingleRawScan(sd->m_positionNumber, !preview);
    }
}

std::vector<ScanPtr> ScanDataManager::getScans()
{
    return m_io.getRawScans(false);
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
