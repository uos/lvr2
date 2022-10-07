//
// Created by praktikum on 13.09.22.
//

#ifndef LAS_VEGAS_SCANPROJECTSCHEMARDBX_HPP
#define LAS_VEGAS_SCANPROJECTSCHEMARDBX_HPP
#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>
#include <dirent.h>
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include <regex>
namespace lvr2
{

    class ScanProjectSchemaRdbx : public DirectorySchema
    {
        public:
        ScanProjectSchemaRdbx(const std::string& rootDir) : DirectorySchema( rootDir), dir_{rootDir}{};
        ~ScanProjectSchemaRdbx() = default;
        virtual Description scanProject() const;



        virtual Description position(
                const size_t& scanPosNo) const;

        virtual Description lidar(
                const size_t& scanPosNo,
                const size_t& lidarNo) const;

        virtual Description camera( const size_t& scanPosNo,
        const size_t& camNo) const;


        virtual Description scan(
                const size_t& scanPosNo,
                const size_t& lidarNo,
                const size_t& scanNo) const;

        virtual Description scanChannel(
                const size_t& scanPosNo,
                const size_t& lidarNo,
                const size_t& scanNo,
                const std::string& channelName) const;

        // virtual std::string scanChannelInv(
        //     const std::string& d_data) const;

        virtual Description cameraImage(
                const size_t& scanPosNo,
                const size_t& camNo,
                const std::vector<size_t>& cameraImageNos) const;

        virtual Description cameraImageGroup(
                const size_t& scanPosNo,
                const size_t& camNo,
                const std::vector<size_t>& cameraImageGroupNos) const;

        virtual Description hyperspectralCamera(
                const size_t& scanPosNo,
                const size_t& camNo) const;

        virtual Description hyperspectralPanorama(
                const size_t& scanPosNo,
                const size_t& camNo,
                const size_t& panoNo) const;

        virtual Description hyperspectralPanoramaPreview(
                const size_t& scanPosNo,
                const size_t& camNo,
                const size_t& panoNo) const;

        virtual Description hyperspectralPanoramaChannel(
                const size_t& scanPosNo,
                const size_t& camNo,
                const size_t& panoNo,
                const size_t& channelNo) const;
        private:
        const std::string& dir_;

    };



} // lvr2

#endif //LAS_VEGAS_SCANPROJECTSCHEMARDBX_HPP
