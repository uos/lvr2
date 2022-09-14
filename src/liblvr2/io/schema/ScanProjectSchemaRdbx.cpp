//
// Created by praktikum on 13.09.22.
//
#include <sstream>
#include <iomanip>

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
        std::stringstream sstr;


        Description dp = scanProject();
        Description d;
        d.dataRoot = *dp.dataRoot + "/final.pose";

        d.metaRoot = d.dataRoot;
        d.meta = "final.pose";

        return d;
    }
    Description ScanProjectSchemaRdbx::lidar(
            const size_t& scanPosNo,
            const size_t& lidarNo) const
    {
        std::stringstream sstr;
        sstr << lidarNo;

        Description dp = position(scanPosNo);

        Description d;
        d.dataRoot = *dp.dataRoot + "/scan" + sstr.str();
        return d;
    }
    Description ScanProjectSchemaRdbx::camera(
            const size_t& scanPosNo,
            const size_t& camNo) const
    {
        std::stringstream sstr;
        sstr << camNo;

        Description dp = position(scanPosNo);

        Description d;
        d.dataRoot = *dp.dataRoot + "/images/" + sstr.str();

        d.metaRoot = d.dataRoot;
        d.meta = "final.pose";


        return d;
    }

    Description ScanProjectSchemaRdbx::scan(
            const size_t& scanPosNo,
            const size_t& lidarNo,
            const size_t& scanNo) const
    {
        std::stringstream sstr;
        sstr <<"scans/"<< scanNo;

        Description dp = lidar(scanPosNo, lidarNo);

        Description d;
        d.dataRoot = *dp.dataRoot + "/" + sstr.str();

        d.metaRoot = d.dataRoot;
        d.meta = "final.pose";

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