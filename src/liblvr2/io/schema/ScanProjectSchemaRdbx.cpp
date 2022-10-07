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
#include  <boost/optional/optional_io.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <fstream>



namespace lvr2 {
    /**
     * scanProject speichert metadaten des Projects
     * @return Description of ScanProject
     */
    Description ScanProjectSchemaRdbx::scanProject() const
    {
        Description d;
        d.dataRoot = "";
        d.metaRoot = d.dataRoot;
        d.meta = "project.json";
        return d;
    }
    /**
     * position speichert metadaten des position
     * @param scanPosNo
     * @return
     */
    Description ScanProjectSchemaRdbx::position(
            const size_t &scanPosNo) const
    {
        Description dp = scanProject();
        Description d;
        //Find Path to Scan Position
        std::stringstream tmp_stream;
        tmp_stream << *dp.dataRoot << "ScanPos" << std::setfill('0') << std::setw(3) << scanPosNo << ".SCNPOS";
        d.dataRoot = tmp_stream.str();

        d.metaRoot = d.dataRoot;
        d.meta = "final.pose";
        return d;
    }
    /**
     *  Lidar Description beinhaltet nur dataRoot um die Ordnung zu erhalten Riegl hat nur einen Scaner
     * @param scanPosNo
     * @param lidarNo
     * @return Lidar Description
     */
    Description ScanProjectSchemaRdbx::lidar(
            const size_t& scanPosNo,
            const size_t& lidarNo) const
    {
        Description d;
        if(lidarNo == 0)
        {
            Description dp = position(scanPosNo);
            d.dataRoot = dp.dataRoot;
        }
        return d;
    }
    /**
    * Camera Description beinhaltet nur dataRoot um die Ordnung zu erhalten Riegl hat nur eine Camera
    * @param scanPosNo
    * @param cameraNo
    * @return camera Discription
    */
    Description ScanProjectSchemaRdbx::camera(
            const size_t& scanPosNo,
            const size_t& camNo) const
    {
        Description d;
        if(camNo == 0)
        {
            Description dp = position(scanPosNo);
            d.dataRoot = dp.dataRoot;
        }
        return d;
    }
    /**
     *
     *  scan Description behinhaltet die Scandata und Metadata
     *
     * @param scanPosNo
     * @param lidarNo
     * @param scanNo
     * @return
     */
    Description ScanProjectSchemaRdbx::scan(
            const size_t& scanPosNo,
            const size_t& lidarNo,
            const size_t& scanNo) const
    {

        Description dp = lidar(scanPosNo,lidarNo);
        auto path = m_rootPath / dp.dataRoot.get() / "scans";
        Description d;

        struct dirent *ent;
        //matching REGEX to Timestamp of Scan
        std::vector<std::string> matching_files;
        boost::filesystem::directory_iterator end_itr;
        std::regex rxRDBX("([0-9]+)\\_([0-9]+)\\.rdbx" );


        for( boost::filesystem::directory_iterator i( path ); i != end_itr; ++i ) {
            // Skip if not a file
            if (!boost::filesystem::is_regular_file(i->status())) continue;
            auto path_str = i->path().filename().string();
            if (regex_match(path_str, rxRDBX)) {
                matching_files.push_back(i->path().stem().string());
            }
        }
        //Sort Files. Rdbx Files come first
        std::sort(matching_files.begin(), matching_files.end());

        if(matching_files.size() > scanNo) {
            d.data = matching_files[scanNo] + ".rdbx";
            d.meta = matching_files[scanNo] + ".scn";
            d.dataRoot= *dp.dataRoot + "/scans";
            d.metaRoot= d.dataRoot;
        }
        return d;
    }

    /**
     * Empty da Riegl keine unterschiedlichen Scanner hat
     * @param scanPosNo
     * @param lidarNo
     * @param scanNo
     * @param channelName
     * @return
     */
    Description ScanProjectSchemaRdbx::scanChannel(
            const size_t& scanPosNo,
            const size_t& lidarNo,
            const size_t& scanNo,
            const std::string& channelName) const
    {
        Description d;
        return d;
    }

    /**
     * cameraImage Description beinhaltet die Fotos und Metadata
     * vergleichbar zu scan
     * @param scanPosNo
     * @param camNo
     * @param cameraImageNos
     * @return
     */
    Description ScanProjectSchemaRdbx::cameraImage(
            const size_t& scanPosNo,
            const size_t& camNo,
            const std::vector<size_t>& cameraImageNos) const
    {

        Description dp = lidar(scanPosNo,camNo);
        auto path = m_rootPath / dp.dataRoot.get() / "images";

        // find the right files
        std::vector<std::string> matching_files;
        boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
        std::regex rxJPG("([0-9]+)\\_([0-9]+)\\_0"+to_string(cameraImageNos.front()+1)+"_([0-9])+\\.jpg" );

        // search the directory
        for( boost::filesystem::directory_iterator i( path ); i != end_itr; ++i ) {
            // Skip if not a file
            if (!boost::filesystem::is_regular_file(i->status())) continue;
            auto path_str = i->path().filename().string();
            if (regex_match(path_str, rxJPG)) {
                matching_files.push_back(i->path().stem().string());
            }
        }

        Description d;
        // Load images and meta data
        if(matching_files.size() > camNo) {
            d.data = matching_files[camNo] + ".jpg";
            d.meta = matching_files[camNo] + ".img";
            d.dataRoot= *dp.dataRoot + "/images";
            d.metaRoot= d.dataRoot;
        }

        return d;
    }

    Description ScanProjectSchemaRdbx::cameraImageGroup(
            const size_t& scanPosNo,
            const size_t& camNo,
            const std::vector<size_t>& cameraImageGroupNos) const
    {
        Description d;
        return d;
    }

    Description ScanProjectSchemaRdbx::hyperspectralCamera(
            const size_t& scanPosNo,
            const size_t& camNo) const
    {
        Description d;
        return d;
    }

    Description ScanProjectSchemaRdbx::hyperspectralPanorama(
            const size_t& scanPosNo,
            const size_t& camNo,
            const size_t& panoNo) const
    {
        Description d;
        return d;
    }

    Description ScanProjectSchemaRdbx::hyperspectralPanoramaPreview(
            const size_t& scanPosNo,
            const size_t& camNo,
            const size_t& panoNo) const
    {
        Description d;
        return d;
    }

    Description ScanProjectSchemaRdbx::hyperspectralPanoramaChannel(
            const size_t& scanPosNo,
            const size_t& camNo,
            const size_t& panoNo,
            const size_t& channelNo
    ) const
    {
        Description d;
        return d;
    }
} // lvr2