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
 * RxpIO.hpp
 *
 *  @date 01.11.2018
 *  @author Alexander Loehr (aloehr@uos.de)
 */

#ifndef LVR2_IO_RXPIO_HPP
#define LVR2_IO_RXPIO_HPP

#include <string>

#include "lvr2/io/BaseIO.hpp"

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/types/MatrixTypes.hpp"
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
        ModelPtr read(std::string filename, int reduction_factor, const Transformd& transform);

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
