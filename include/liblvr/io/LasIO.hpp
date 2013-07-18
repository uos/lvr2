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
 * LasIO.h
 *
 *  @date 03.01.2012
 *  @author Thomas Wiemann
 */

#ifndef LASIO_H_
#define LASIO_H_

#include "BaseIO.hpp"

namespace lvr
{

/**
 * @brief   Interface class to read laser scan data in .las-Format
 */
class LasIO : public BaseIO
{
public:
    LasIO() {};
    virtual ~LasIO() {};

    /**
     * @brief Parse the given file and load supported elements.
     *
     * @param filename  The file to read.
     */
    virtual ModelPtr read(string filename );

    /**
     * @brief Save the loaded elements to the given file.
     *
     * @param filename Filename of the file to write.
     */
    virtual void save( string filename );

};

} /* namespace lvr */
#endif /* LASIO_H_ */
