#include <memory>

#include <ctime>

struct ScanMetaData
{
    std::chrono::system_clock::time_point m_start;
    std::chrono::system_clock::time_point m_end;
    double m_pose[6];
    double m_scanAngles[6]; // TODO Document
};

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
    static int readScanMetaData(boost::filesystem::path& fn, std::shared_ptr<ScanMetaData>& data);
    static int readSpectralMetaData(boost::filesystem::path& fn, std::shared_ptr<SpectralMetaData>& data);
};

} // lvr2
