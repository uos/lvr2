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
 * BoctreeIO.hpp
 *
 *  @date 23.08.2012
 *  @author Thomas Wiemann
 */

#ifndef BOCTREEIO_HPP_
#define BOCTREEIO_HPP_

#include <boost/filesystem.hpp>
#include "slam6d/scan_io_oct.h"
#include "BaseIO.hpp"
#include "geometry/Matrix4.hpp"


namespace lssr
{

/**
 * @brief IO-Class to import compressed octrees from slam6d
 */
class BoctreeIO : public BaseIO
{
public:

    BoctreeIO();
    virtual ~BoctreeIO();

    /**
     * \brief Parse the given file and load supported elements.
     *
     * @param filename  The file to read.
     */
    virtual ModelPtr read(string filename );


    /**
     * \brief Save the loaded elements to the given file.
     *
     * @param filename Filename of the file to write.
     */
    virtual void save( string filename );

private:
    Matrix4<float> parseFrameFile(ifstream& frameFile);
};

} /* namespace lssr */
#endif /* BOCTREEIO_HPP_ */
