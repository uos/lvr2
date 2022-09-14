//
// Created by praktikum on 13.09.22.
//
#include <sstream>
#include <iomanip>
#include <boost/filesystem/operations.hpp>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRdbx.hpp"


namespace lvr2 {

    Description ScanProjectSchemaRdbx::scanProject() const
    {
        Description d;

        d.dataRoot = "scans";
        d.metaRoot = d.dataRoot;
        d.meta = "final.pose";

        return d;
    }
    Description ScanProjectSchemaRdbx::position(
            const size_t &scanPosNo) const
    {


        Description dp = scanProject();
        Description d;
        d.dataRoot = *dp.dataRoot + "ScanPos" + scanPosNo + "SCNPOS";
        d.metaRoot = d.dataRoot;
        d.meta = "final.pose";

        return d;
    }


    Description ScanProjectSchemaRdbx::lidar(
            const size_t& scanPosNo,
            const size_t& lidarNo) const
    {

        DIR *dir;
        Description dp = position(scanPosNo);
        char *path;
        *path = dp.dataRoot;
        Description d;
        d.dataRoot= *dp.dataRoot + "scans";
        d.metaRoot= *dp.dataRoot + "scans";
        struct dirent *ent;
        std::regex rxRDBX("([0-9]+)\\_([0-9]+)\\.rdbx" );
        std::regex rxSCN("([0-9]+)\\_([0-9]+)\\.SCN" );

        if ((dir = opendir (path)) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL) {
                if (regex_match((ent->d_name), rxRDBX)) {
                    d.data = ent->d_name;
                }
                if (regex_match((ent->d_name), rxSCN)) {
                    d.meta = ent->d_name;

                }
            }
            closedir (dir);
        } else {
            /* could not open directory */
            perror ("");
            return EXIT_FAILURE;
        }
        return d;
    }


    Description ScanProjectSchemaRdbx::camera(
            const size_t& scanPosNo,
            const size_t& camNo) const
    {
        DIR *dir;
        Description dp = position(scanPosNo);
        char *path;
        *path = dp.dataRoot;
        Description d;
        d.dataRoot= *dp.dataRoot + "images";
        d.metaRoot= dp.dataRoot;
        d.meta=dp.meta;
        struct dirent *ent;
        std::regex rxJPG("([0-9]+)\\_([0-9]+)\\" + camNo + ".jpg" );


        if ((dir = opendir (path)) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL) {
                if (regex_match((ent->d_name), rxJPG)) {
                    d.data = ent->d_name;
                }

            }
            closedir (dir);
        } else {
            /* could not open directory */
            perror ("");
            return EXIT_FAILURE;
        }
        return d;


    }

    Description ScanProjectSchemaRdbx::scan(
            const size_t& scanPosNo,
            const size_t& lidarNo,
            const size_t& scanNo) const
    {


        Description d;


        return d;
    }
    Description ScanProjectSchemaRdbx::scanChannel(
            const size_t& scanPosNo,
            const size_t& lidarNo,
            const size_t& scanNo,
            const std::string& channelName) const
    {
        Description d;



        return d;
    }

    //Unterschied zu Camera ????
    Description ScanProjectSchemaRdbx::cameraImage(
            const size_t& scanPosNo,
            const size_t& camNo,
            const std::vector<size_t>& cameraImageNos) const
    {


        Description d;


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