#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Logging.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRiSCANVZ600.hpp"


#include <sstream>
#include <iomanip>

#include <filesystem>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
namespace lvr2
{

ScanProjectSchemaRiSCANVZ600::ScanProjectSchemaRiSCANVZ600(const std::string& rootDir) : DirectorySchema(rootDir), m_rootDir(rootDir)
{
    boost::property_tree::ptree project_tree;
    boost::property_tree::read_xml(rootDir + "/project.rsp", project_tree);

    auto scanPositions = project_tree.get_child_optional("project.scanpositions");
    if(scanPositions)
    {
        auto positionTree = *scanPositions;
        for(auto &scanposition : positionTree)
        {
            std::string positionName = scanposition.second.get("<xmlattr>.name", "");
            std::string scanImageName = scanposition.second.get("scanposimages.scanposimage.<xmlattr>.name", "");
            // skip empty nodes
            if (positionName.empty())
            {
                continue;
            }
            // skip invalid scans
            if (scanImageName.empty())
            {
                lvr2::logout::get() << lvr2::warning << "Scan " << positionName << " is invalid, skipping" << lvr2::endl;
                continue;
            }

            std::cout << positionName << std::endl;
            std::cout << scanImageName << std::endl;
            m_scanPositions.push_back(positionName);

        }

        
    }
    else
    {
        lvr2::logout::get() << lvr2::warning << "Could not find scanpositions in project.rsp" << lvr2::endl;
    }
}

Description ScanProjectSchemaRiSCANVZ600::scanProject() const
{
    Description d;
    d.dataRoot = "";
    d.metaRoot = d.dataRoot;
    d.meta = "project.json"; // temporarily copied over from raw .PROJ
    return d;
}

Description ScanProjectSchemaRiSCANVZ600::position(
    const size_t &scanPosNo) const
{
    Description dp = scanProject();
    Description d;
    // Find Path to Scan Position
    std::stringstream tmp_stream;
    tmp_stream << *dp.dataRoot << "Scans/ScanPos" << std::setfill('0') << std::setw(3) << scanPosNo;
    d.dataRoot = tmp_stream.str();


    d.metaRoot = d.dataRoot;
    d.meta = "POSEESTIMATIONS/final.pose.pose";
    return d;
}

Description ScanProjectSchemaRiSCANVZ600::lidar(
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

Description ScanProjectSchemaRiSCANVZ600::camera(
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

Description ScanProjectSchemaRiSCANVZ600::scan(
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

Description ScanProjectSchemaRiSCANVZ600::scanChannel(
    const size_t &scanPosNo,
    const size_t &lidarNo,
    const size_t &scanNo,
    const std::string &channelName) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRiSCANVZ600::cameraImage(
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
    // std::regex rxJPG("([0-9]+)\\_([0-9]+)\\_0" + std::to_string(cameraImageNo + 1) + "\\_0"+std::to_string(groupNo + 1)+"\\.jpg");
    std::regex rxJPG("([0-9]+)\\_([0-9]+)"
        + (groupNo < 9 ? std::string("\\_0") : std::string("\\_"))
        + std::to_string(groupNo + 1)
        + "\\_01" // filler (always 01 fwict)
        + "\\_c" + std::to_string(cameraImageNo + 1)
        + "\\.jpg");

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

Description ScanProjectSchemaRiSCANVZ600::hyperspectralCamera(
    const size_t &scanPosNo,
    const size_t &camNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRiSCANVZ600::hyperspectralPanorama(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &panoNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRiSCANVZ600::hyperspectralPanoramaPreview(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &panoNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRiSCANVZ600::hyperspectralPanoramaChannel(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &panoNo,
    const size_t &channelNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaRiSCANVZ600::cameraImageGroup(
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