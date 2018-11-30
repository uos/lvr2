/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
