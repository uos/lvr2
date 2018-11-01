/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * ScanprojectIO.hpp
 *
 *  @date 01.11.2018
 *  @author Alexander Loehr (aloehr@uos.de)
 */

#ifndef LVR2_IO_SCANPROJECTIO_HPP
#define LVR2_IO_SCANPROJECTIO_HPP

#include <lvr2/io/BaseIO.hpp>
#include <lvr2/geometry/Matrix4.hpp>
#include <lvr2/geometry/BaseVector.hpp>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

namespace lvr2
{

using Vec = BaseVector<float>;

/**
 * @brief A struct that holds information for an image
 */
struct ImageFile
{
    /// transformation matrix
    Matrix4<Vec> orientation_transform;
    /// transformation matrix
    Matrix4<Vec> extrinsic_transform;

    /// path to image
    fs::path image_file;

    /// intrinsic parameters in this order: fx, fy, Cx, Cy
    float intrinsic_params[4];

    /// distortion params in this order: k1, k2, k3, k4, p1, p2
    float distortion_params[6];
};

/**
 * @brief A struct that holds information for a scan position.
 */
struct ScanPosition
{
    /// file path for pointcloud data
    fs::path               scan_file;
    /// transformation  matrix from scan position space to scan project space
    Matrix4<Vec>           transform;
    /// a vector with image informations for this scanposition
    std::vector<ImageFile> images;
};

/**
 * @brief A struct that holds information for an UOS Scanproject
 */
struct Scanproject
{
    /// directory with calibration information
    fs::path calib_dir;
    /// directory with image data
    fs::path img_dir;
    /// directory with scan data
    fs::path scans_dir;
    /// A vector with ScanPositions that are part of this scan project
    std::vector<ScanPosition> scans;
};



/**
 * @brief Class for reading/writing and parsing a UOS Scanproject directory.
 */
class ScanprojectIO : public BaseIO
{
    public:

        /**
         * @brief Reads the giving directory as an UOS Scanproject directory and
         * return a ModelPtr with the read data.
         *
         * @param dir the path to the UOS Scanproject directory
         *
         * @return a ModelPtr with the data read from dir
         */
        ModelPtr read(std::string dir);

        /**
         * @brief Saving UOS Scanprojects is currently not supported.
         */
        void save(std::string dir);

        /**
         * @brief Parses a directory as an UOS Scanproject
         *
         * @param dir The directory path
         *
         * @param silent Suppresses any error messages while parsing the directory. This is usefull
         *               if you only want to figure out if the directory is a valid UOS Scanproject
         *               directory and aren't interested in any warnings if it isn't one.
         *
         * @return Returns true if it successfully parsed an UOS Scanproject and elsewise false.
         */
        bool parse_project(const std::string& dir, bool silent = false);

        /**
         * @brief Returns the parsed UOS Scanproject
         *
         * @return Returns the parsed UOS Scanproject in an Scanproject struct
         *         if parse_project(...) was executed before and returned true
         *         else it will return an empty Scanproject struct.
         */
        Scanproject& get_project();

    private:

        /// @cond internal
        template<typename ValueType>
        bool load_params_from_file(ValueType *buf, const fs::path &src, unsigned int count);
        bool exists_and_is_dir(const fs::path &dir, bool silent);
        fs::path project_dir;
        Scanproject project;
        /// @endcond internal
};

} // namespace lvr2

#endif
