#include "lvr2/io/ScanDataManager.hpp"

namespace lvr2
{

ScanDataManager::ScanDataManager(std::string filename) : m_io(filename)
{
    
}

void ScanDataManager::loadPointCloudData(ScanPtr& sd, bool preview)
{
    if ((!sd->pointsLoaded && !preview) || ( sd->pointsLoaded && preview))
    {
        sd = m_io.getSingleRawScan(sd->positionNumber, !preview);
    }
}

std::vector<ScanPtr> ScanDataManager::getScans()
{
    return m_io.getRawScans(false);
}

std::vector<std::vector<ScanImage> > ScanDataManager::getCameraData()
{
    return m_io.getRawCamData(false);
}

cv::Mat ScanDataManager::loadImageData(int scan_id, int cam_id)
{
    ScanImage ret = m_io.getSingleRawCamData(scan_id, cam_id, true);
    return ret.image;
}


} // namespace lvr2
