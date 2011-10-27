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
 * PPMIO.h
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

#ifndef PPMIO_HPP_
#define PPMIO_HPP_

namespace lssr
{

/**
 * @brief An implementation of the PPM file format.
 */
template<typename ColorType>
class PPMIO
{
public:
    PPMIO();

    void write(string filename);
    void setDataArray(ColorType** array, size_t sizeX, size_t sizeY);

private:
    ColorType**              m_data;

    size_t                  m_sizeX;
    size_t                  m_sizeY;

};

}

#include "PPMIO.tcc"

#endif /* PPMIO_H_ */
