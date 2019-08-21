#include "lvr2/types/Scan.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/AsciiIO.hpp"
#include "lvr2/io/IOUtils.hpp"

#include <boost/filesystem.hpp>

#include <iostream>

namespace lvr2
{

void parseSLAMDirectory(std::string dir, vector<ScanPtr>& scans)
{
    boost::filesystem::path directory(dir);
    if(is_directory(directory))
    {

        boost::filesystem::directory_iterator lastFile;
        std::vector<boost::filesystem::path> scan_files;

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
                    scan_files.push_back(p);
                }
            }
        }

        if(scan_files.size() > 0)
        {
            for(size_t i = 0; i < scan_files.size(); i++)
            {
                ScanPtr scan = ScanPtr(new Scan());

                std::string filename = (scan_files[i]).stem().string();
                boost::filesystem::path frame_file(filename + ".frames");
                boost::filesystem::path pose_file(filename + ".pose");

                boost::filesystem::path frame_path = directory/frame_file;
                boost::filesystem::path pose_path = directory/pose_file;

                std::cout << "Loading '" << filename << "'" << std::endl;
                AsciiIO io;
                ModelPtr model = io.read(scan_files[i].string());
                scan->m_points = model->m_pointCloud;

                size_t numPoints = scan->m_points->numPoints();
                floatArr pts = scan->m_points->getPointArray();

                for (size_t i = 0; i < numPoints; i++)
                {
                    BaseVector<float> pt(pts[i*3 + 0], pts[i*3 + 1], pts[i*3 + 2]);
                    scan->m_boundingBox.expand(pt);
                }

                Transformd pose_estimate = Transformd::Identity();
                Transformd registration = Transformd::Identity();

                if(boost::filesystem::exists(frame_path))
                {
                    std::cout << timestamp << "Loading frame information from " << frame_path << std::endl;
                    registration = getTransformationFromFrames<double>(frame_path);
                }
                else
                {
                    std::cout << timestamp << "Did not find a frame file for " << filename << std::endl;
                }

                if(boost::filesystem::exists(pose_path))
                {
                    std::cout << timestamp << "Loading pose estimation from " << pose_path << std::endl;
                    pose_estimate = getTransformationFromPose<double>(pose_path);
                }
                else
                {
                    std::cout << timestamp << "Did not find a pose file for " << filename << std::endl;
                }

                // transform points?
                scan->m_registration = registration;
                scan->m_poseEstimation = pose_estimate;

                scans.push_back(scan);
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
