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

namespace lssr
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
         * @brief Reads the given file and stores point and normal
         *        information in the given parameters
         *
         * @param filename      The file to read
         */
        virtual ModelPtr read( string filename );


        /**
         * @todo : Implement save method for ASCII Files...
         * @param filename
         */
        virtual void save( string filename );


        /**
         * \brief Save model to given filename.
         * 
         * This method will save the model ti the given filename. Additional one single line comment can be set.
         *
         * \param filename  File to write.
         * \param comment   Single line comment.
         **/
        void save( std::string filename, std::string comment );


        /**
         * \brief Save model to file.
         *
         * Save a model. Additional options can be set via option multimap.
         *
         * \param filename  Filename of the output file.
         * \param options   Additional options.
         * \param m         Model to save.
         **/
        void save( string filename,
                std::multimap< std::string, std::string > options, 
                ModelPtr m = ModelPtr() );


        /// TODO: Coordinate mapping for ascii files
        static size_t countLines(string filename);


        /**
         * @brief Helper method. Returns the number of columns in the
         *        given file.
         */
        static int getEntriesInLine(string filename);



};


} // namespace lssr


#endif /* ASCIIIO_H_ */
