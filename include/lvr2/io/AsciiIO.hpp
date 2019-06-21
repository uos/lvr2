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

 /**
 *
 * @file      AsciiIO.hpp
 * @brief     Read and write pointclouds from .pts and .3d files.
 * @details   Read and write pointclouds from .pts and .3d files.
 * 
 * @author    Thomas Wiemann (twiemann), twiemann@uos.de, Universität Osnabrück
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   111001
 * @date      Created:       2011-03-09
 * @date      Last modified: 2011-10-01 19:49:24
 *
 **/

#ifndef ASCIIIO_H_
#define ASCIIIO_H_

#include "lvr2/io/BaseIO.hpp"

namespace lvr2
{

/**
 * @brief A import / export class for point cloud data in plain
 *        text formats. Currently the file extensions .xyz, .txt,
 *       .3d and .pts are supported.
 */
class AsciiIO : public BaseIO
{
    public:

        /**
         * \brief Default constructor.
         **/
        AsciiIO() {};


        /**
         * @brief Reads the given file and stores point and color
         *        information in the given parameters
         *
         * @param filename      The file to read
         */
        virtual ModelPtr read( string filename );

        /**
         * @brief read  Parses the given file and stores point and color attribute
         *              information in the returned model pointer. It is assumed that
         *              the each line in the file (except the first one which may be a
         *              header line) contains point coordinates (x, y, z) and optional
         *              color and intensity information. The position (columns) of the
         *              supported attritbutes within each line are define by the
         *              respective parameters given to this function. Each line may
         *              consist of more attributes, but only the ones specified are
         *              parsed. Not existing attributes are indicated by -1.
         *
         * @param filename  The file to parse
         * @param x         The colum number containing the x-coordinate of a point
         * @param y         The colum number containing the y-coordinate of a point
         * @param z         The colum number containing the z-coordinate of a point
         * @param r         The colum number containing the r color component (or -1)
         * @param g         The colum number containing the g color component (or -1)
         * @param b         The colum number containing the b color component (or -1)
         * @param i         The colum number containing the intensity value (or -1)
         * @return
         */
        virtual ModelPtr read(
                string filename,
                const int& x, const int& y, const int& z,
                const int& r = -1, const int& g = -1, const int& b = -1, const int& i = -1);


        /**
         * @todo : Implement save method for ASCII Files...
         * @param filename
         */
        virtual void save( string filename );


        /// TODO: Coordinate mapping for ascii files
        static size_t countLines(string filename);


        /**
         * @brief Helper method. Returns the number of columns in the
         *        given file.
         */
        static int getEntriesInLine(string filename);
};


} // namespace lvr2

#endif /* ASCIIIO_H_ */
