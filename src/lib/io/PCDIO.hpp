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
 * @file      PCDIO.hpp
 * @brief     Read and write point clouds from PCD files.
 * @details   Read and write point clouds from the files in the
 *            PointCloudLibrary file format.
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   120109
 * @date      Created:       2012-01-09 01:50:19
 * @date      Last modified: 2012-01-09 01:50:22
 *
 **/

#ifndef PCDIO_HPP_INCLUDED
#define PCDIO_HPP_INCLUDED

#include "BaseIO.hpp"

namespace lssr
{

/**
 * @brief A import / export class for point cloud data in the PointCloudLibrary
 *        file format.
 */
class PCDIO : public BaseIO
{
    public:

        /**
         * \brief Default constructor.
         **/
        PCDIO() {};


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


};


} // namespace lssr


#endif /* PCDIO_HPP_INCLUDED */
