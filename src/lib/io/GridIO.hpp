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
 * GridIO.hpp
 *
 *  @date 10.01.2012
 *  @author Thomas Wiemann
 */

#ifndef GRIDIO_HPP_
#define GRIDIO_HPP_

#include "DataStruct.hpp"

#include <string>

namespace lssr
{

class GridIO
{
public:
    GridIO();
    void read( std::string filename );
    virtual ~GridIO();

    floatArr getPoints( size_t &n );
    uintArr  getBoxes(  size_t &n );

private:
    floatArr m_points;
    uintArr  m_boxes;
    size_t   m_numPoints;
    size_t   m_numBoxes;
};

} /* namespace lssr */


#endif /* GRIDIO_HPP_ */
