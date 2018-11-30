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

#ifndef LVR2_PPMIO_HPP_
#define LVR2_PPMIO_HPP_

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

using std::string;
using std::ifstream;
using std::stringstream;
using std::cout;
using std::endl;

namespace lvr2
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

} // namespace lvr2

#endif /* PPMIO_H_ */
