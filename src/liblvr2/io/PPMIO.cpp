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
 * PPMIO.cpp
 *
 *  Created on:  08.09.2011
 *      Author:  Kim Rinnewitz  ( krinnewitz@uos.de )
 *  Modified on: 11.12.2011
 *      Author:  Thomas Wiemann ( twiemann@uos.de )
 *  Modified on: 15.02.2011
 *      Author:  Denis Meyer    ( denmeyer@uos.de )
 */

#include "lvr2/io/PPMIO.hpp"

#include <iostream>
#include <fstream>
#include <string.h>

using namespace std;

namespace lvr2
{

PPMIO::PPMIO()
{
    m_pixels    = 0;
    m_width   = 0;
    m_height   = 0;
}

PPMIO::PPMIO( string filename ) : m_width(0), m_height(0), m_pixels(0)
{

    // Try to open file
    ifstream in(filename.c_str());

    // Parse file
    if(in.good())
    {
        // Line buffer
        char buffer[1024];

        // Read file tag
        readLine(in, buffer);

        // Check tag
        string tag(buffer);
        if(tag == "P3")
        {
            // Read width, height and color information
            stringstream ss;
            readLine(in, buffer);
            ss << buffer << " ";
            readLine(in, buffer);
            ss << buffer << " ";

            // Read formatted data
            ss >> m_width >> m_height;

            // Alloc data
            m_pixels = new unsigned char[m_width * m_height * 3];

            // Read pixels
            int p;
            for(int i = 0; i < m_width * m_height * 3; i++)
            {
                in >> p;
                m_pixels[i] = (unsigned char)p;
            }
        }
        else
        {
            in.close();
            in.open(filename.c_str(), ios::binary);
            //	    readLine(in, buffer);
            //	    char tmp[3];
            //	    sscanf(buffer, "%s %d %d 255", tmp, &m_width, &m_height);

            string tag;
            in >> tag;


            if(tag == "P6") // TODO: hacked in for our output
            {
                int n_colors;
                in >> m_width >> m_height >> n_colors;
		in.getline(0,0);
                m_pixels = new unsigned char[m_width * m_height * 3];
                in.read((char *)m_pixels, m_width * m_height * 3);
            }
            else
            {
                cerr << "Unsupported tag, only P3 or P6 possible." << endl;
            }
        }
    }
    else
    {
        cout << "ReadPPM: Unable to open file " << filename << "." << endl;
    }
}

void PPMIO::write( string filename )
{
    ofstream out(filename.c_str());

    if(out.good())
    {
    	out<<"P6"<<" "<<m_width<<" "<<m_height<<" "<<"255"<<endl;
	out.write((char*) m_pixels, m_width * m_height * 3);
    }

    out.close();

}

void PPMIO::setDataArray( unsigned char* array, int width, int height )
{
    m_pixels = array;
    m_width = width;
    m_height = height;
}

void PPMIO::readLine( ifstream & in, char* buffer )
{
    // Read lines until no comment line was found
    do
    {
      in.getline(buffer, 256);
    }
    while(buffer[0] == '#' && in.good() );
}

} // namespace lvr2
