#include <memory>
#include <chrono>
#include <vector>


// boost
#include <boost/filesystem.hpp>

#include "lvr2/io/ScanData.hpp"

//struct ScanMetaData
//{
//    std::chrono::system_clock::time_point m_start;
//    std::chrono::system_clock::time_point m_end;
//    double m_pose[6];
//    double m_scanAngles[6]; // TODO Document
//};
//
struct SpectralMetaData
{
    size_t m_numImages;
    SpectralMetaData(size_t numImages) : m_numImages(numImages)
    {
        m_timeStamps.reserve(m_numImages);
        m_angles.reserve(m_numImages);
    }

    std::vector<std::chrono::system_clock::time_point> m_timeStamps;
    std::vector<double> m_angles;
};


namespace lvr2{
class PlutoMetaDataIO{
  public:
      /**
       * @brief 
       *        parse Pose in m_poseEstimation 4x4 Matrix.
       *        parse angles in fov?
       *
       * @param fn
       * @param data
       *    
       * @return 
       */
    static int readScanMetaData(boost::filesystem::path& fn, ScanData& data);

    static int readSpectralMetaData(boost::filesystem::path& fn, std::shared_ptr<SpectralMetaData>& data);
};

} // lvr2
