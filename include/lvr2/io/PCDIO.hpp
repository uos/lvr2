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

#include "lvr2/io/BaseIO.hpp"

namespace lvr2
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


} // namespace lvr2

#endif /* PCDIO_HPP_INCLUDED */
