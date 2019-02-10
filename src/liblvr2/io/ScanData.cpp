#include "lvr2/io/ScanData.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/AsciiIO.hpp"
#include "lvr2/io/IOUtils.hpp"

#include <boost/filesystem.hpp>

#include <iostream>

namespace lvr2
{

void parseSLAMDirectory(std::string dir, vector<ScanData>& scans)
{
    boost::filesystem::path directory(dir);
    if(is_directory(directory))
    {

        boost::filesystem::directory_iterator lastFile;
        std::vector<boost::filesystem::path> scan_data_files;

        // First, look for .3d files
        for(boost::filesystem::directory_iterator it(directory); it != lastFile; it++ )
        {
            boost::filesystem::path p = it->path();
            if(p.extension().string() == ".3d")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(p.filename().string().c_str(), "scan%3d", &num))
                {
                    scan_data_files.push_back(p);
                }

            }
        }

        if(scan_data_files.size() > 0)
        {
            for(size_t i = 0; i < scan_data_files.size(); i++)
            {
                ScanData scan_data;

                std::string filename = (scan_data_files[i]).stem().string();
                boost::filesystem::path frame_file(filename + ".frames");
                boost::filesystem::path pose_file(filename + ".pose");

                boost::filesystem::path frame_path = directory/frame_file;
                boost::filesystem::path pose_path = directory/pose_file;

                std::cout << "Loading '" << filename << "'" << std::endl;
                AsciiIO io;
                ModelPtr model = io.read(scan_data_files[i].string());
                scan_data.m_points = model->m_pointCloud;

                size_t numPoints = scan_data.m_points->numPoints();
                floatArr pts = scan_data.m_points->getPointArray();

                for (size_t i = 0; i < numPoints; i++)
                {
                    Vector<BaseVector<float> > pt(pts[i*3 + 0], pts[i*3 + 1], pts[i*3 + 2]);
                    scan_data.m_boundingBox.expand(pt);
                }

                Eigen::Matrix4d pose_estimate = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d registration = Eigen::Matrix4d::Identity();

                if(boost::filesystem::exists(frame_path))
                {
                    std::cout << timestamp << "Loading frame information from " << frame_path << std::endl;
                    registration = getTransformationFromFrames(frame_path);
                }
                else
                {
                    std::cout << timestamp << "Did not find a frame file for " << filename << std::endl;
                }

                if(boost::filesystem::exists(frame_path))
                {
                    std::cout << timestamp << "Loading pose estimation from " << pose_path << std::endl;
                    pose_estimate = getTransformationFromPose(pose_path);
                }
                else
                {
                    std::cout << timestamp << "Did not find a pose file for " << filename << std::endl;
                }

                scan_data.m_registration = registration;
                scan_data.m_poseEstimation = pose_estimate;

                scans.push_back(scan_data);
            }
        }
        else
        {
            std::cout << timestamp << "Error in parseSLAMDirectory(): '"
                      << "Directory does not contain any .3d files." << std::endl;
        }
    }
    else
    {
        std::cout << timestamp << "Error in parseSLAMDirectory(): '"
                  << dir << "' is nor a directory." << std::endl;
    }
}

    } // namespace lvr2
