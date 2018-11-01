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
 * RxpIO.hpp
 *
 *  @date 01.11.2018
 *  @author Alexander Loehr (aloehr@uos.de)
 */

#ifndef LVR2_IO_RXPIO_HPP
#define LVR2_IO_RXPIO_HPP

#include <string>

#include <lvr2/io/BaseIO.hpp>

#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Matrix4.hpp>

namespace lvr2
{

using Vec = BaseVector<float>;

/**
 * @brief Reads .rxp files
 */
class RxpIO : public BaseIO
{
    public:

        /**
         * @brief Reads .rxp files
         *
         * @param filename the path to file that should be read
         *
         * @return returns a ModelPtr with the pointcloud data or if it was unable to read the file
         *         it returns an empty ModelPtr
         */
        ModelPtr read(std::string filename);

        /**
         * @brief Reads .rxp files
         *
         * @param filename the path to file that should be read
         *
         * @param reduction_factor only saves a point if (point_number % reduction_factor == 0).
         *                         So it only saves every reduction_factor th point. If reduction
         *                         factor is 1, every point will be saved and if it is smaller
         *                         than 1 it will be set to 1 automatically.
         *
         * @param transform every Point will be transformed with the given tranformation matrix
         *                  before it is put into the pointcloud
         *
         * @return returns a ModelPtr with the pointcloud data or if it was unable to read the file
         *         it returns an empty ModelPtr
         */
        ModelPtr read(std::string filename, int reduction_factor, Matrix4<Vec> transform);

        /**
         * @brief This function is not supported and will do nothing.
         */
        void save(std::string filename);

    private:

        /// @cond internal
        bool check_error(int error_code);
        /// @endcond
};

} // namespace lvr2

#endif
