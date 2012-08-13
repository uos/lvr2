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
 *  Created on:  08.09.2011
 *      Author:  Kim Rinnewitz  ( krinnewitz@uos.de )
 *  Modified on: 11.12.2011
 *      Author:  Thomas Wiemann ( twiemann@uos.de )
 *  Modified on: 15.02.2011
 *      Author:  Denis Meyer    ( denmeyer@uos.de )
 */

#ifndef PPMIO_HPP_
#define PPMIO_HPP_

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

using std::string;
using std::ifstream;
using std::stringstream;
using std::cout;
using std::endl;

namespace lssr
{

/**
 * @brief An implementation of the PPM file format.
 */
class PPMIO
{
public:
    PPMIO();
    PPMIO( string filename );
    virtual ~PPMIO() {};

    void write( string filename );
    void setDataArray( unsigned char* array, int width, int height );

    int            getHeight() const { return m_height; }
    int            getWidth()  const { return m_width;  }
    unsigned char* getPixels() const { return m_pixels; }

private:
    int            m_width;  // The width of the image
    int            m_height; // The height of the image
    unsigned char* m_pixels; // The image/pixel data

    /**
     * Reads a new line from the given stream that is no comment
     * @param   in      The stream to read from
     * @param   buffer  The extracted information
     */
    void readLine( ifstream & in, char* buffer );
};

}

#endif /* PPMIO_H_ */
