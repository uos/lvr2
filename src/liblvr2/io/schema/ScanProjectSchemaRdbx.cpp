//
// Created by praktikum on 13.09.22.
//
#include <sstream>
#include <iomanip>
#include <boost/filesystem/operations.hpp>
#include <filesystem>
#include <vector>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRdbx.hpp"
#include <boost/filesystem.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <fstream>

namespace lvr2
{

Description ScanProjectSchemaRdbx::scanProject() const
{
    Description d;
    d.dataRoot = "";
    d.metaRoot = d.dataRoot;
    d.meta = "project.json";
    return d;
}

Description ScanProjectSchemaRdbx::position(
    const size_t &scanPosNo) const
{
    Description dp = scanProject();
    Description d;
    // Find Path to Scan Position
    std::stringstream tmp_stream;
    tmp_stream << *dp.dataRoot << "ScanPos" << std::setfill('0') << std::setw(3) << scanPosNo << ".SCNPOS";
    d.dataRoot = tmp_stream.str();

    d.metaRoot = d.dataRoot;
    d.meta = "final.pose";
    return d;
}

Description ScanProjectSchemaRdbx::lidar(
    const size_t &scanPosNo,
    const size_t &lidarNo) const
{
    Description d;
    if (lidarNo == 0)
    {
        Description dp = position(scanPosNo);
        d.dataRoot = dp.dataRoot;
    }
    return d;
}

Description ScanProjectSchemaRdbx::camera(
    const size_t &scanPosNo,
    const size_t &camNo) const
{
    Description d;
    if (camNo == 0)
    {
        Description dp = position(scanPosNo);
        d.dataRoot = dp.dataRoot;
    }
    return d;
}

Description ScanProjectSchemaRdbx::scan(
    const size_t &scanPosNo,
    const size_t &lidarNo,
    const size_t &scanNo) const
{

    Description dp = lidar(scanPosNo, lidarNo);
    auto path = m_rootPath / dp.dataRoot.get() / "scans";
    Description d;

    struct dirent *ent;
    // matching REGEX to Timestamp of Scan
    std::vector<std::string> matching_files;
    boost::filesystem::directory_iterator end_itr;
    std::regex rxRDBX("([0-9]+)\\_([0-9]+)\\.rdbx");

    for (boost::filesystem::directory_iterator i(path); i != end_itr; ++i)
    {
        // Skip if not a file
        if (!boost::filesystem::is_regular_file(i->status()))
            continue;
        auto path_str = i->path().filename().string();
        if (regex_match(path_str, rxRDBX))
        {
            matching_files.push_back(i->path().stem().string());
        }
    }
    // Sort Files. Rdbx Files come first
    std::sort(matching_files.begin(), matching_files.end());

    if (matching_files.size() > scanNo)
    {
        d.data = matching_files[scanNo] + ".rdbx";
        d.meta = matching_files[scanNo] + ".scn";
        d.dataRoot = *dp.dataRoot + "/scans";
        d.metaRoot = d.dataRoot;
    }
    return d;
}

Description ScanProjectSchemaRdbx::scanChannel(
    const size_t &scanPosNo,
    const size_t &lidarNo,
    const size_t &scanNo,
    const std::string &channelName) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRdbx::cameraImage(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &groupNo,
    const size_t &cameraImageNo) const
{
    Description dp = lidar(scanPosNo, camNo);
    auto path = m_rootPath / dp.dataRoot.get() / "images";

    // Find the correct files
    std::vector<std::string> matching_files;
    boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
    //std::regex rxJPG("([0-9]+)\\_([0-9]+)\\_0" + std::to_string(cameraImageNo + 1) + "_([0-9])+\\.jpg");
    std::regex rxJPG("([0-9]+)\\_([0-9]+)\\_0" + std::to_string(cameraImageNo + 1) + "\\_0"+std::to_string(groupNo + 1)+"\\.jpg");      
    // Search the directory
    for (boost::filesystem::directory_iterator i(path); i != end_itr; ++i)
    {
        // Skip if not a file
        if (!boost::filesystem::is_regular_file(i->status()))
        {
            continue;
        }

        auto path_str = i->path().filename().string();

        if(i->path().extension() == ".jpg")
        {
            if (regex_match(path_str, rxJPG))
            {
                //std::cout << "Found match: " << groupNo << " " << cameraImageNo << " -> " << path_str << std::endl;
                matching_files.push_back(i->path().stem().string());
            }
            else
            {
                //std::cout << "No match: " << groupNo << " " << cameraImageNo << " -> " << path_str << std::endl;
            }
        }
    }
    
    std::cout << matching_files.size() << " " << groupNo << " " << cameraImageNo << std::endl;

    Description d;
    // Load images and meta data
    if (matching_files.size() > camNo)
    {
        d.data = matching_files[camNo] + ".jpg";
        d.meta = matching_files[camNo] + ".img";
        d.dataRoot = *dp.dataRoot + "/images";
        d.metaRoot = d.dataRoot;
    }

    return d;
}

Description ScanProjectSchemaRdbx::hyperspectralCamera(
    const size_t &scanPosNo,
    const size_t &camNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRdbx::hyperspectralPanorama(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &panoNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRdbx::hyperspectralPanoramaPreview(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &panoNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRdbx::hyperspectralPanoramaChannel(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &panoNo,
    const size_t &channelNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRdbx::cameraImageGroup(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &GroupNo) const
{
    // Riegl only has one group (at least we only support one right now)
    // If we want support multiple image sets here, we need to extend
    // the schema here to search for the correct images
    if(GroupNo < 1)
    {
        // Same as camera without meta
        Description d = camera(scanPosNo, camNo);
        d.metaRoot = boost::none;
        d.meta = boost::none;
        return d;
    }
    else
    {
        return Description();
    }
    
}
} // lvr2