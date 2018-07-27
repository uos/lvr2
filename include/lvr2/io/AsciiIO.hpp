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

#include "BaseIO.hpp"

namespace lvr2
{

/**
 * @brief A import / export class for point cloud data in plain
 *        text formats. Currently the file extensions .xyz, .txt,
 *        .3d and .pts are supported.
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


} // namespace lvr


#endif /* ASCIIIO_H_ */
