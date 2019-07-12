#include <iomanip>
#include <sstream>
#include <boost/format.hpp>

#include "lvr2/io/ScanDirectoryParser.hpp"
#include "lvr2/io/IOUtils.hpp"

using namespace boost::filesystem;

namespace lvr2
{

ScanDirectoryParser::ScanDirectoryParser(const std::string& directory) noexcept
{
    // Check if directory exists and save path
    Path dir(directory);
    if(!exists(directory))
    {
        std::cout << timestamp << "Directory " << directory << " does not exist." << std::endl;
    }
    else
    {
        m_directory = directory;
    }

    // Set default prefixes and extension
    m_pointExtension = ".txt";
    m_poseExtension = ".dat";
    m_pointPrefix = "scan";
    m_posePrefix = "scan";

    m_start = 0;
    m_end = 0;
}

void ScanDirectoryParser::setPointCloudPrefix(const std::string& prefix)
{
    m_pointPrefix = prefix;
}
void ScanDirectoryParser::setPointCloudExtension(const std::string& extension)
{
    m_pointExtension = extension;
}
void ScanDirectoryParser::setPosePrefix(const std::string& prefix)
{
    m_posePrefix = prefix;
}   
void ScanDirectoryParser::setPoseExtension(const std::string& extension)
{
    m_poseExtension = extension;
}

void ScanDirectoryParser::setStart(int s)
{
    m_start = s;
}
void ScanDirectoryParser::setEnd(int e)
{
    m_end = e;
}

void ScanDirectoryParser::setTargetSize(const size_t& size)
{
    m_targetSize = size;
}

size_t ScanDirectoryParser::examinePLY(const std::string filename)
{
    return getNumberOfPointsInPLY(filename);
}

size_t ScanDirectoryParser::examineASCII(const std::string filename)
{
    Path p(filename);
    return countPointsInFile(p);
} 

size_t ScanDirectoryParser::computeNumberOfPoints()
{
    size_t n = 0;
    for(size_t i = m_start; i <= m_end; i++)
    {
        // Construct name of current file
        std::stringstream point_ss;
        point_ss << m_pointPrefix << boost::format("%03d") % i << m_pointExtension;
        std::string pointFileName = point_ss.str();
        Path pointPath = Path(m_directory)/Path(pointFileName);

        // Check for file and get number of points
        if(exists(pointPath))
        {
            std::cout << timestamp << "Counting points in file " << pointPath << std::endl;
            if(pointPath.extension() == ".3d" || pointPath.extension() == ".txt" || pointPath.extension() == ".pts")
            {
                n += examineASCII(pointPath.string());
            }
            else if(pointPath.extension() == ".ply")
            {
                n += examinePLY(pointPath.string());
            }
        }
        else
        {
            std::cout << timestamp << "File " << pointPath << " does not exist." << std::endl;
        }
    }
    return n;
}

 

} // namespace lvr2