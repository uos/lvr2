#include <iomanip>
#include <sstream>
#include <boost/format.hpp>

#include "lvr2/io/ScanDirectoryParser.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/registration/OctreeReduction.hpp"

using namespace boost::filesystem;

namespace lvr2
{

ScanDirectoryParser::~ScanDirectoryParser()
{
    // Delete scan descriptions
    for(auto i : m_scans)
    {
        delete i;
    }
}

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
    cout << extension << endl;
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


size_t ScanDirectoryParser::examinePLY(const std::string& filename)
{
    return getNumberOfPointsInPLY(filename);
}

size_t ScanDirectoryParser::examineASCII(const std::string& filename)
{
    Path p(filename);
    return countPointsInFile(p);
} 

PointBufferPtr ScanDirectoryParser::octreeSubSample(const double& voxelSize, const size_t& minPoints)
{
    ModelPtr out_model(new Model);

    for(auto i : m_scans)
    {
        std::cout << timestamp << "Reading " << i->m_filename << std::endl;
        ModelPtr model = ModelFactory::readModel(i->m_filename);
        if(model)
        {
            PointBufferPtr buffer = model->m_pointCloud;
            if(buffer)
            {
                std::cout << timestamp << "Building octree with voxel size " << voxelSize << " from " << i->m_filename << std::endl;
                OctreeReduction oct(buffer, voxelSize, 5);
                PointBufferPtr reduced = oct.getReducedPoints();

                // Apply transformation
                std::cout << timestamp << "Transforming reduced point cloud" << std::endl;
                out_model->m_pointCloud = reduced;
                transformPointCloud<double>(out_model, i->m_pose);

                // Write reduced data
                std::stringstream name_stream;
                Path p(i->m_filename);
                name_stream << p.stem().string() << "_reduced" << ".ply";
                std::cout << timestamp << "Saving data to " << name_stream.str() << std::endl;
                ModelFactory::saveModel(out_model, name_stream.str());

                std::cout << timestamp << "Points written: " << reduced->numPoints() << std::endl;
            }
        }
    }
    return PointBufferPtr(new PointBuffer);
}

PointBufferPtr ScanDirectoryParser::randomSubSample(const size_t& tz)
{
    ModelPtr out_model(new Model);

    // Compute global reduction ratio and clamp
    size_t actual_points = 0;

    for(auto i : m_scans)
    {
        ModelPtr model = ModelFactory::readModel(i->m_filename);
        if(model)
        {
            PointBufferPtr buffer = model->m_pointCloud;
            if(buffer)
            {
                PointBufferPtr reduced = 0;
                int target_size = 0;
                if(tz > 0)
                {
                    // Calc number of points to sample
                    float total_ratio = (float)i->m_numPoints / m_numPoints;
                    float target_ratio = total_ratio * tz;


                    target_size = (int)(target_ratio + 0.5);
                    std::cout << timestamp << "Sampling " << target_size << " points from " << i->m_filename << std::endl;

                    // Sub-sample buffer
                    reduced = subSamplePointBuffer(buffer, target_size);
                }
                else
                {
                    std::cout << timestamp << "Using orignal points from " << i->m_filename << std::endl;
                    reduced = buffer;
                    target_size = buffer->numPoints();
                }

                // Apply transformation
                std::cout << timestamp << "Transforming point cloud" << std::endl;
                out_model->m_pointCloud = reduced;
                transformPointCloud<double>(out_model, i->m_pose);

                // Write reduced data
                std::stringstream name_stream;
                Path p(i->m_filename);
                name_stream << p.stem().string() << "_reduced" << ".ply";
                std::cout << timestamp << "Saving data to " << name_stream.str() << std::endl;
                ModelFactory::saveModel(out_model, name_stream.str());

                actual_points += target_size;
                std::cout << timestamp << "Points written: " << actual_points << " / " << tz << std::endl;
            }
        }
    }
    return out_model->m_pointCloud;
}

void ScanDirectoryParser::parseDirectory()
{
    std::cout << timestamp << "Parsing directory" << m_directory << std::endl;
    std::cout << timestamp << "Point prefix and extension: " << m_pointPrefix << " " << m_pointExtension << std::endl;
    std::cout << timestamp << "Pose prefix and extension: " << m_posePrefix << " " << m_poseExtension << std::endl;

    m_numPoints = 0;
    for(size_t i = m_start; i <= m_end; i++)
    {
        // Construct name of current file
        std::stringstream point_ss;
        point_ss << m_pointPrefix << boost::format("%03d") % i << m_pointExtension;
        std::string pointFileName = point_ss.str();
        Path pointPath = Path(m_directory)/Path(pointFileName);

        // Construct name of transformation file
        std::stringstream pose_ss;
        pose_ss << m_posePrefix << boost::format("%03d") % i << m_poseExtension;
        std::string poseFileName = pose_ss.str();
        Path posePath = Path(m_directory)/Path(poseFileName);

        // Check for file and get number of points
        size_t n = 0;
        if(exists(pointPath))
        {
            std::cout << timestamp << "Counting points in file " << pointPath << std::endl;
            if(pointPath.extension() == ".3d" || pointPath.extension() == ".txt" || pointPath.extension() == ".pts")
            {
                n = examineASCII(pointPath.string());
            }
            else if(pointPath.extension() == ".ply")
            {
                n = examinePLY(pointPath.string());
            }
            m_numPoints += n;
        }
        else
        {
            std::cout << timestamp << "Point cloud file " << pointPath << " does not exist." << std::endl;
        }

        // Check for pose information file
        Transformd matrix = Transformd::Identity();

        if(exists(posePath))
        {
            matrix = getTransformationFromFile<double>(posePath.string());
            std::cout << timestamp << "Found transformation: " << posePath << " @ " << std::endl << matrix << std::endl;
        }
        else
        {
            std::cout << timestamp << "Scan pose file " << posePath << "does not exist. Will not transfrom." << std::endl;
        }


        
        ScanInfo* info = new ScanInfo;
        info->m_filename     = std::string(pointPath.string());
        info->m_numPoints    = n;
        info->m_pose         = matrix;

        m_scans.push_back(info);
    }
    std::cout << timestamp << "Finished parsing. Directory contains " << m_scans.size() << " scans with " << m_numPoints << " points." << std::endl;
}

 

} // namespace lvr2
